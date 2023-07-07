import torch
import torch.nn as nn
import time
import numpy as np
from datetime import datetime


class MyFit(nn.Module):
    def __init__(self, model, optimizer, scheduler, writer, loss, device, fout):
        super(MyFit, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.loss_fn = loss
        self.device = device
        self.fout = fout

    def log_string(self, out_str):
        self.fout.write(out_str+'\n')
        self.fout.flush()
        print(out_str)

    def train_one_epoch(self, trainloader, epoch_index):
        running_loss = 0.
        last_loss = 0.
        running_acc = 0.0
        num_pred = 0.0
        num_total = 0.0

        start_time = time.time()
        i = 0
        for batch in trainloader:
            if batch is None:
                continue

            batch = [value.to(self.device) for value in batch]
            mesh, target = batch[:-1], batch[-1].to(torch.long)
            torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            pred_logits = self.model(*mesh)
            valid_mask = (target>=0) &(target<pred_logits.shape[-1])
            loss = self.loss_fn(pred_logits[valid_mask], target[valid_mask])
            loss.backward()
            self.optimizer.step()

            pred_labels = torch.argmax(pred_logits.detach(), dim=-1)
            matched = (pred_labels[valid_mask]==target[valid_mask])
            acc = matched.to(torch.float)
            running_acc += torch.mean(acc).item()

            num_pred += torch.sum(acc)
            num_total += acc.shape[0]

            # Gather data and report
            running_loss += loss.detach().item()
            if i%50==49:
                last_loss = running_loss/(i+1)  # loss per batch
                last_runtime = (time.time()-start_time)/(i+1)
                self.log_string('train batch {} loss: {:.4f}, accuracy: {:.0f}/{:.0f}={:.2f}, '
                                'runtime-per-batch: {:.2f} ms'.format(i+1, last_loss, num_pred, num_total,
                                                               num_pred/num_total*100, last_runtime*1000))
                self.writer.add_scalar('Loss/train', last_loss, epoch_index)
                self.writer.add_scalar('Accuracy/train', num_pred/num_total*100, epoch_index)

            i = i+1
        avg_acc = running_acc/i
        return last_loss, avg_acc

    def evaluate(self, testloader, report_iou=False, class_names=None):
        running_loss = 0.
        running_acc = 0.0
        num_pred = 0.0
        num_total = 0.0

        all_gt_labels = []
        all_pred_labels = []
        start_time = time.time()
        i = 0
        for batch in testloader:
            if batch is None:
                continue

            batch = [value.to(self.device) for value in batch]
            mesh, target = batch[:-1], batch[-1].to(torch.long)
            torch.cuda.empty_cache()

            pred_logits = self.model(*mesh)
            valid_mask = (target >= 0) & (target < pred_logits.shape[-1])
            loss = self.loss_fn(pred_logits[valid_mask], target[valid_mask])
            pred_logits = pred_logits.detach()
            pred_labels = torch.argmax(pred_logits, dim=-1)

            running_loss += loss.detach().item()
            matched = (pred_labels[valid_mask]==target[valid_mask])
            acc = matched.to(torch.float)
            running_acc += torch.mean(acc).item()

            num_pred += torch.sum(acc)
            num_total += acc.shape[0]

            if report_iou:
                all_gt_labels.append(target[valid_mask].cpu().numpy())
                all_pred_labels.append(pred_labels[valid_mask].cpu().numpy())
            i = i+1

        avg_runtime = (time.time()-start_time)/i
        avg_tloss = running_loss/i
        avg_tacc = running_acc/i
        self.log_string('test loss: {:.4f}, test accuracy:{:.2f}, runtime-per-mesh: {:.2f} ms'
                        .format(avg_tloss, avg_tacc*100, avg_runtime*1000))

        if report_iou:
            all_gt_labels = np.concatenate(all_gt_labels, axis=0)
            all_pred_labels = np.concatenate(all_pred_labels, axis=0)
            self.evaluate_iou(all_gt_labels, all_pred_labels, class_names)
        return avg_tloss, avg_tacc

    def evaluate_iou(self, gt_labels, pred_labels, class_names):
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
        self.log_string('eval overall class accuracy:\t %d/%d=%3.2f' % (total_correct, total_seen,
                                                                             100 * total_correct / float(total_seen)))
        self.log_string('eval average class accuracy:\t %3.2f' % (100 * np.mean(list(class_acc.values()))))
        for cat in class_names:
            self.log_string('eval mIoU of %14s:\t %3.2f' % (cat, 100 * class_iou[cat]))
        self.log_string('eval mIoU of all %d classes:\t %3.2f'%(len(class_names), 100*np.mean(list(class_iou.values()))))

    def __call__(self, ckpt_epoch, num_epochs, trainloader, testloader, write_dir,
                 report_iou=False, class_names=None):
        self.writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], 0)

        best_tacc = 0
        for epoch in range(ckpt_epoch, num_epochs):
            self.log_string("************************Epoch %03d Training********************"%(epoch+1))
            self.log_string(str(datetime.now()))
            self.model.train(True)
            avg_loss, avg_acc = self.train_one_epoch(trainloader, epoch)
            self.scheduler.step()

            self.log_string("=======================Epoch %03d Evaluation===================="%(epoch+1))
            self.log_string(str(datetime.now()))
            self.model.train(False)
            avg_tloss, avg_tacc = self.evaluate(testloader, report_iou, class_names)
            self.log_string("****************************************************************\n")

            self.writer.add_scalars('Loss', {'Train': avg_loss, 'Test': avg_tloss}, epoch+1)
            self.writer.add_scalars('Accuracy', {'Train': avg_acc, 'Test': avg_tacc}, epoch+1)
            self.writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], epoch+1)

            # Track best performance, and save the model's state
            # if avg_tacc > best_tacc:
            #     best_tacc = avg_tacc
            model_path = '{}/model_epoch_{}'.format(write_dir, epoch+1)
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                        model_path, _use_new_zipfile_serialization=False)

        self.writer.close()
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                     f"{write_dir}/best_model", _use_new_zipfile_serialization=False)
        return