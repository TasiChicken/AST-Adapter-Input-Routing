import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs

def get_routing_aux_weight(
    epoch,
    total_epochs,
    max_weight=0.01,
    warmup_epochs=5,
    decay_start=20,
    decay_end=50,
):
    """
    Auxiliary loss schedule.

    Early stage:
        encourage routing exploration / avoid collapse.

    Late stage:
        decay to 0 so that classification loss dominates.
    """
    if max_weight <= 0:
        return 0.0

    if epoch < warmup_epochs:
        return max_weight * float(epoch + 1) / float(max(warmup_epochs, 1))

    if epoch < decay_start:
        return max_weight

    if epoch < decay_end:
        progress = float(epoch - decay_start) / float(max(decay_end - decay_start, 1))
        return max_weight * 0.5 * (1.0 + math.cos(math.pi * progress))

    return 0.0

def get_routing_explore_eps(epoch, max_eps=0.0, decay_end=50):
    """
    Cosine decay exploration probability.

    epoch 0: max_eps
    epoch >= decay_end: 0
    """
    if max_eps <= 0:
        return 0.0

    if epoch >= decay_end:
        return 0.0

    progress = float(epoch) / float(max(decay_end, 1))
    return max_eps * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_routing_explore_eps(model, eps):
    """
    Set explore_eps for all GumbelNetwork modules.
    """
    model_to_check = model.module if hasattr(model, "module") else model

    for module in model_to_check.modules():
        if hasattr(module, "explore_eps"):
            module.explore_eps = float(eps)

def collect_routing_probs(model):
    """
    Collect last_prob from each GumbelNetwork after model forward.

    Returns:
        probs: list of tensors, each [B, K]
    """
    probs = []
    model_to_check = model.module if hasattr(model, "module") else model

    for blk in model_to_check.blocks:
        for module in blk.modules():
            if hasattr(module, "last_prob"):
                probs.append(module.last_prob)

    return probs

def routing_auxiliary_loss(
    probs,
    loss_type="anti_collapse",
    collapse_threshold=0.90,
    entropy_weight=0.0,
    eps=1e-8,
):
    """
    loss_type:
        none:
            no auxiliary loss.

        anti_collapse:
            only penalize when one branch dominates the batch.
            Conservative, but may be too weak.

        balance:
            L2 loss between batch-average routing and uniform distribution.

        kl_balance:
            KL divergence between batch-average routing and uniform distribution.
            Usually smoother than L2.

        kl_entropy:
            KL balance + per-sample entropy warmup.
            More effective for preventing early one-hot collapse.
    """
    if len(probs) == 0 or loss_type == "none":
        return None

    losses = []

    for p in probs:
        # p: [B, K]
        avg_p = p.mean(dim=0)  # [K]
        k = avg_p.numel()
        target = torch.ones_like(avg_p) / k

        if loss_type == "anti_collapse":
            max_usage = avg_p.max()
            loss = torch.relu(max_usage - collapse_threshold) ** 2

        elif loss_type == "balance":
            loss = ((avg_p - target) ** 2).sum()

        elif loss_type == "kl_balance":
            loss = (
                avg_p * (
                    torch.log(avg_p + eps) -
                    torch.log(target + eps)
                )
            ).sum()

        elif loss_type == "kl_entropy":
            kl_loss = (
                avg_p * (
                    torch.log(avg_p + eps) -
                    torch.log(target + eps)
                )
            ).sum()

            # Encourage per-sample routing not to become one-hot too early.
            entropy = -(p * torch.log(p + eps)).sum(dim=-1)
            entropy = entropy / math.log(k)
            entropy_loss = 1.0 - entropy.mean()

            loss = kl_loss + entropy_weight * entropy_loss

        else:
            raise ValueError(f"Unknown routing auxiliary loss type: {loss_type}")

        losses.append(loss)

    return sum(losses) / len(losses)   

def _get_branch_names_from_block(blk, num_branches):
    """
    Get branch names from block if available.
    Works for sp_tp_relu and sp_tp.
    """
    if hasattr(blk, "inter_key") and blk.inter_key:
        return list(blk.inter_key)

    if hasattr(blk, "inter_key_a") and blk.inter_key_a:
        return list(blk.inter_key_a)

    default_names = ["sp", "tp", "relu"]
    return default_names[:num_branches]


@torch.no_grad()
def update_soft_route_stats(model, soft_route_stats, device):
    """
    Accumulate soft routing probability statistics over all batches.
    Uses last_prob from each GumbelNetwork after forward.
    """
    model_to_check = model.module if hasattr(model, "module") else model

    for layer_id, blk in enumerate(model_to_check.blocks):
        probs = []

        for module in blk.modules():
            if hasattr(module, "last_prob"):
                probs.append(module.last_prob.detach())

        if len(probs) == 0:
            continue

        p = probs[-1]  # [B, K]
        p_sum = p.sum(dim=0).to(device)
        count = torch.tensor(float(p.size(0)), device=device)

        branch_names = _get_branch_names_from_block(blk, p.size(-1))

        if layer_id not in soft_route_stats:
            soft_route_stats[layer_id] = {
                "sum": torch.zeros_like(p_sum, device=device),
                "count": torch.tensor(0.0, device=device),
                "branch_names": branch_names,
            }

        soft_route_stats[layer_id]["sum"] += p_sum
        soft_route_stats[layer_id]["count"] += count


@torch.no_grad()
def sync_soft_route_stats(soft_route_stats, device):
    """
    Synchronize soft routing probability statistics across DDP ranks.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for layer_id in soft_route_stats:
            soft_route_stats[layer_id]["sum"] = soft_route_stats[layer_id]["sum"].to(device)
            soft_route_stats[layer_id]["count"] = soft_route_stats[layer_id]["count"].to(device)

            torch.distributed.all_reduce(
                soft_route_stats[layer_id]["sum"],
                op=torch.distributed.ReduceOp.SUM,
            )
            torch.distributed.all_reduce(
                soft_route_stats[layer_id]["count"],
                op=torch.distributed.ReduceOp.SUM,
            )

    return soft_route_stats


@torch.no_grad()
def print_soft_route_stats(soft_route_stats, title="[Soft Routing Probability Statistics]"):
    """
    Print averaged softmax probability over all accumulated batches.
    This is different from inter_idx / argmax statistics.
    """
    if len(soft_route_stats) == 0:
        print(f"\n{title}")
        print("No soft routing statistics found.")
        return

    print(f"\n{title}")

    for layer_id in sorted(soft_route_stats.keys()):
        stat = soft_route_stats[layer_id]
        p = stat["sum"] / stat["count"].clamp(min=1.0)
        p = p.detach().cpu()

        branch_names = stat.get("branch_names", ["sp", "tp", "relu"][:p.numel()])

        ratio_str = ", ".join([
            f"{branch_names[i]}={p[i].item():.4f}"
            for i in range(min(len(branch_names), p.numel()))
        ])

        print(f"Layer {layer_id:02d}: {ratio_str}")

@torch.no_grad()
def update_route_stats(model, route_stats, device):
    """
    Update hard / argmax routing statistics for the current batch.
    This reads blk.inter_idx, so it is branch-choice statistics, not soft probability.
    """
    model_to_check = model.module if hasattr(model, "module") else model

    for layer_id, blk in enumerate(model_to_check.blocks):
        if hasattr(blk, "inter_idx"):
            idx = blk.inter_idx.detach().to(device).long()  # [B]

            if hasattr(blk, "inter_key") and blk.inter_key:
                branch_names = list(blk.inter_key)
                num_branches = len(branch_names)
            else:
                num_branches = int(idx.max().item()) + 1 if idx.numel() > 0 else 3
                branch_names = ["sp", "tp", "relu"][:num_branches]

            if layer_id not in route_stats:
                route_stats[layer_id] = {
                    "counts": torch.zeros(
                        num_branches,
                        dtype=torch.long,
                        device=device,
                    ),
                    "branch_names": branch_names,
                }

            counts = torch.bincount(idx, minlength=num_branches)
            route_stats[layer_id]["counts"] += counts

@torch.no_grad()
def sync_route_stats(route_stats, device):
    """
    Synchronize hard / argmax routing statistics across all DDP ranks.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for layer_id in route_stats:
            route_stats[layer_id]["counts"] = route_stats[layer_id]["counts"].to(device)
            torch.distributed.all_reduce(
                route_stats[layer_id]["counts"],
                op=torch.distributed.ReduceOp.SUM,
            )

    return route_stats

@torch.no_grad()
def print_route_stats(route_stats, title="[AST-Adapter Routing Statistics]"):
    """
    Print hard / argmax routing statistics.
    """
    if len(route_stats) == 0:
        print(f"\n{title}")
        print("No routing statistics found.")
        return

    print(f"\n{title}")

    for layer_id in sorted(route_stats.keys()):
        counts = route_stats[layer_id]["counts"].detach().cpu().float()
        ratio = counts / counts.sum().clamp(min=1)

        branch_names = route_stats[layer_id].get(
            "branch_names",
            ["sp", "tp", "relu"][:counts.numel()],
        )

        ratio_str = ", ".join([
            f"{branch_names[i]}={ratio[i].item():.3f}"
            for i in range(min(len(branch_names), counts.numel()))
        ])

        count_str = ", ".join([
            f"{branch_names[i]}={int(counts[i].item())}"
            for i in range(min(len(branch_names), counts.numel()))
        ])

        print(f"Layer {layer_id:02d}: {ratio_str} | counts: {count_str}")

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, num_training_steps_per_epoch=None,
                    update_freq=None,
                    routing_aux_loss="none",
                    routing_aux_weight=0.0,
                    routing_aux_warmup_epochs=5,
                    routing_aux_decay_start=20,
                    routing_aux_decay_end=50,
                    routing_collapse_threshold=0.90,
                    total_epochs=90,
                    routing_explore_eps=0.0,
                    routing_explore_decay_end=50,
                    routing_entropy_weight=0.0,):
    model.train(True)
    current_explore_eps = get_routing_explore_eps(
        epoch=epoch,
        max_eps=routing_explore_eps,
        decay_end=routing_explore_decay_end,
    )

    set_routing_explore_eps(model, current_explore_eps)

    print(f"Routing explore eps = {current_explore_eps:.6f}")
    metric_logger = utils.MetricLogger(delimiter="  ")
    route_stats = {}
    soft_route_stats = {}
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('routing_aux', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('routing_aux_w', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)

            routing_aux_value = torch.tensor(0.0, device=device)
            lambda_routing = get_routing_aux_weight(
                epoch=epoch,
                total_epochs=total_epochs,
                max_weight=routing_aux_weight,
                warmup_epochs=routing_aux_warmup_epochs,
                decay_start=routing_aux_decay_start,
                decay_end=routing_aux_decay_end,
            )

            if routing_aux_loss != "none" and lambda_routing > 0:
                routing_probs = collect_routing_probs(model)
                aux_loss = routing_auxiliary_loss(
                    routing_probs,
                    loss_type=routing_aux_loss,
                    collapse_threshold=routing_collapse_threshold,
                    entropy_weight=routing_entropy_weight,
                )
                if aux_loss is not None:
                    routing_aux_value = aux_loss.detach()
                    loss = loss + lambda_routing * aux_loss

        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

                routing_aux_value = torch.tensor(0.0, device=device)
                lambda_routing = get_routing_aux_weight(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    max_weight=routing_aux_weight,
                    warmup_epochs=routing_aux_warmup_epochs,
                    decay_start=routing_aux_decay_start,
                    decay_end=routing_aux_decay_end,
                )

                if routing_aux_loss != "none" and lambda_routing > 0:
                    routing_probs = collect_routing_probs(model)
                    aux_loss = routing_auxiliary_loss(
                        routing_probs,
                        loss_type=routing_aux_loss,
                        collapse_threshold=routing_collapse_threshold,
                        entropy_weight=routing_entropy_weight,
                    )
                    if aux_loss is not None:
                        routing_aux_value = aux_loss.detach()
                        loss = loss + lambda_routing * aux_loss

        update_route_stats(model, route_stats, device)
        update_soft_route_stats(model, soft_route_stats, device)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(routing_aux=routing_aux_value.item())
        metric_logger.update(routing_aux_w=lambda_routing)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(routing_aux=routing_aux_value.item(), head="loss")
            log_writer.update(routing_aux_w=lambda_routing, head="loss")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # gather routing stats from all DDP ranks
    route_stats = sync_route_stats(route_stats, device)
    soft_route_stats = sync_soft_route_stats(soft_route_stats, device)

    print("Averaged stats:", metric_logger)

    if utils.is_main_process():
        print_route_stats(route_stats, title=f"[Train AST-Adapter Routing Statistics][Epoch {epoch}]")
        print_soft_route_stats(
            soft_route_stats,
            title=f"[Train Soft Routing Probability Statistics][Epoch {epoch}]",
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    route_stats = {}
    soft_route_stats = {}

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)

            # update routing statistics after forward
            update_route_stats(model, route_stats, device)
            update_soft_route_stats(model, soft_route_stats, device)

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # gather routing stats from all DDP ranks
    route_stats = sync_route_stats(route_stats, device)
    soft_route_stats = sync_soft_route_stats(soft_route_stats, device)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # print only on main process
    if utils.is_main_process():
        print_route_stats(route_stats, title="[Val AST-Adapter Routing Statistics]")
        print_soft_route_stats(soft_route_stats, title="[Val Soft Routing Probability Statistics]")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    route_stats = {}
    soft_route_stats = {}
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        # AST-Adapter routing statistics
        update_route_stats(model, route_stats, device)
        update_soft_route_stats(model, soft_route_stats, device)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        open(file, 'a').close()
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # gather routing stats from all DDP ranks
    route_stats = sync_route_stats(route_stats, device)
    soft_route_stats = sync_soft_route_stats(soft_route_stats, device)

    # Print AST-Adapter routing statistics only on main process
    if utils.is_main_process():
        print_route_stats(route_stats, title="[Test AST-Adapter Routing Statistics]")
        print_soft_route_stats(
            soft_route_stats,
            title="[Test Soft Routing Probability Statistics]",
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            left = line.rfind('[')
            right = line.rfind(']')
            if left < 0 or right < 0 or right <= left:
                print("[merge skip] cannot find logits brackets:", repr(line[:300]))
                continue

            name = line[:left].strip()
            tail = line[right + 1:].strip().split()
            if len(tail) < 3:
                print("[merge skip] invalid tail:", repr(line[:300]))
                continue

            label = tail[0]
            chunk_nb = tail[1]
            split_nb = tail[2]

            data_str = line[left + 1:right]
            data = np.fromstring(data_str, dtype=np.float64, sep=',')
            if data.size == 0:
                print("[merge skip] empty logits:", repr(line[:300]))
                continue

            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
