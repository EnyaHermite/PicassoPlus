import torch
import time
import numpy as np


def evaluate(model, loss_fn, mesh_files, label_files, transform_fn, class_names, device):
    running_loss = 0.
    running_acc = 0.0
    num_pred = 0.0
    num_total = 0.0

    model.eval()
    all_gt_labels = []
    all_pred_labels = []
    start_time = time.time()
    for i in range(len(mesh_files)):
        mesh_path = mesh_files[i]
        label_path = label_files[i]

        values = transform_fn(mesh_path, label_path)
        values = list(values)
        for k in range(len(values)):
            values[k] = values[k].to(device)
        torch.cuda.empty_cache()

        mesh, target, rev_ids = values[:-2], values[-2].to(torch.long), values[-1].to(torch.long)
        pred_logits = model(*mesh)

        pred_logits = pred_logits.detach()
        pred_logits = pred_logits[rev_ids]
        pred_labels = torch.argmax(pred_logits, dim=-1)

        valid_mask = (target>=0) &(target<pred_logits.shape[-1])
        loss = loss_fn(pred_logits[valid_mask], target[valid_mask])
        running_loss += loss.item()

        matched = (pred_labels[valid_mask]==target[valid_mask])
        acc = matched.to(torch.float)
        running_acc += torch.mean(acc).item()

        num_pred += torch.sum(acc)
        num_total += acc.shape[0]

        all_gt_labels.append(target[valid_mask])
        all_pred_labels.append(pred_labels[valid_mask])

    avg_runtime = (time.time()-start_time)/(i+1)
    avg_tloss = running_loss/(i+1)
    avg_tacc = running_acc/(i+1)
    print('test loss: {:.4f}, test accuracy:{:.2f}, runtime-per-mesh: {:.2f} ms'
           .format(avg_tloss, avg_tacc*100, avg_runtime*1000))

    all_gt_labels = torch.concat(all_gt_labels, dim=0)
    all_pred_labels = torch.concat(all_pred_labels, dim=0)
    all_gt_labels = all_gt_labels.cpu().numpy()
    all_pred_labels = all_pred_labels.cpu().numpy()
    evaluate_iou(all_gt_labels, all_pred_labels, class_names)
    return avg_tloss, avg_tacc


def evaluate_iou(gt_labels, pred_labels, class_names):
    total_seen_class = {cat: 0 for cat in class_names}
    total_correct_class = {cat: 0 for cat in class_names}
    total_union_class = {cat: 0 for cat in class_names}

    for l, cat in enumerate(class_names):
        total_seen_class[cat] += np.sum(gt_labels == l)
        total_union_class[cat] += (np.sum((pred_labels == l) | (gt_labels == l)))
        total_correct_class[cat] += (np.sum((pred_labels == l) & (gt_labels == l)))

    class_iou = {cat: 0.0 for cat in class_names}
    class_acc = {cat: 0.0 for cat in class_names}
    for cat in class_names:
        class_iou[cat] = total_correct_class[cat] / (float(total_union_class[cat]) + np.finfo(float).eps)
        class_acc[cat] = total_correct_class[cat] / (float(total_seen_class[cat]) + np.finfo(float).eps)

    total_correct = sum(list(total_correct_class.values()))
    total_seen = sum(list(total_seen_class.values()))
    print('eval overall class accuracy:\t %d/%d=%3.2f' % (total_correct, total_seen,
                                                          100*total_correct/float(total_seen)))
    print('eval average class accuracy:\t %3.2f' % (100 * np.mean(list(class_acc.values()))))
    for cat in class_names:
        print('eval mIoU of %14s:\t %3.2f' % (cat, 100 * class_iou[cat]))
    print('eval mIoU of all %d classes:\t %3.2f'%(len(class_names), 100*np.mean(list(class_iou.values()))))