from tqdm import tqdm
import numpy as np
import torch
import os, time
from utils import AverageMeter, evaluate, EmptyWith, log_print
from consistency_loss import cosine_similarity_fun


class Trainer(object):
    def __init__(self,
        train_loader=None, 
        test_loader=None,
        model=None, 
        optimizer=None, 
        loss_fn=None, 
        consistency_fn=None,
        consistency_rate=1.0,
        log_interval=100, 
        best_recond={"acc":0,"auc":0,"epoch":-1},
        save_dir="ckpt/test",
        exp_name="test",
        amp=False):
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.model=model
        self.optimizer=optimizer
        self.loss_fn=loss_fn
        self.consistency_fn=consistency_fn
        self.consistency_rate=consistency_rate
        self.log_interval=log_interval
        self.best_record = best_recond
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.amp = amp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)

    def train_epoch(self,epoch):
        self.model.train()

        train_loss_ce = AverageMeter()
        train_loss_consistency = AverageMeter()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_auc = AverageMeter()
        feature_norm = AverageMeter()

        start_time = time.time()

        # scaler
        if self.amp:
            scaler = torch.cuda.amp.GradScaler()

        for batch_idx,(data,label) in enumerate(self.train_loader): 
            if type(data) is list:
                data = torch.cat(data,dim=0)
                label = torch.cat([label,label],dim=0)
            # [64, 3, 299, 299] [64]
            data = data.to(self.device)
            label = label.to(self.device)
            N = label.size(0)
            # forward
            self.optimizer.zero_grad()
            if self.amp:
                amp_class = torch.cuda.amp.autocast
            else:
                amp_class = EmptyWith

            with amp_class():
                # forward and loss
                # [64, 2048] [64,2]
                feature, outputs = self.model(data)
                feature_norm.update(torch.mean(torch.sqrt(torch.sum(feature*feature,dim=1))).item(),N)
                loss_ce = self.loss_fn(outputs,label)
                if self.consistency_fn is not None:
                    loss_consistency = self.consistency_fn(feature)
                    loss = self.consistency_rate * loss_consistency + loss_ce
                else:
                    loss = loss_ce
            # backward
            if self.amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            outputs = outputs.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc, auc, tdr = evaluate(outputs,label)
            train_loss_ce.update(loss_ce.item(), N)
            if self.consistency_fn is not None:
                train_loss_consistency.update(loss_consistency.item(), N)
            train_loss.update(loss.item(), N)
            train_acc.update(acc, N)
            train_auc.update(auc, N)

            if (batch_idx+1) % self.log_interval == 0:
                if self.consistency_fn is not None:
                    msg = '[{}][train] [epoch {}], [iter {} / {}], [loss {:.8f}],[loss ce{:.8f}],[loss consistency {:.8f}], [acc {:.5f}], [auc {:.5f}], [time used {:.1f}], [time left {:.1f}], [feature norm {}]'.format(
                    self.exp_name, epoch, batch_idx, len(self.train_loader), train_loss.avg,train_loss_ce.avg,train_loss_consistency.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, (time.time()-start_time)/60/(batch_idx+1)*(len(self.train_loader)-batch_idx-1), feature_norm.avg)
                else:
                    msg = '[{}][train] [epoch {}], [iter {} / {}], [loss {:.8f}], [acc {:.5f}], [auc {:.5f}], [time used {:.0f}], [time left {:.0f}], [feature norm {}]'.format(
                    self.exp_name, epoch, batch_idx, len(self.train_loader), train_loss.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, (time.time()-start_time)/60/(batch_idx+1)*(len(self.train_loader)-batch_idx-1), feature_norm.avg)
                log_print(msg)
        if self.consistency_fn is not None:
            msg = '[{}][train] [epoch {}], [loss {:.8f}], [loss ce {:.8f}],[loss consistency {:.8f}], [acc {:.5f}], [auc {:.5f}], [time {:.0f}], [lr {:.5f}], [feature norm {}]'.format(
            self.exp_name, epoch, train_loss.avg, train_loss_ce.avg,train_loss_consistency.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, self.optimizer.param_groups[0]['lr'], feature_norm.avg)
        else:
            msg = '[{}][train] [epoch {}], [loss {:.8f}], [acc {:.5f}], [auc {:.5f}], [time {:.0f}], [lr {:.5f}], [feature norm {}]'.format(
            self.exp_name, epoch, train_loss.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, self.optimizer.param_groups[0]['lr'], feature_norm.avg)
        log_print(msg)

    def train_epoch_text(self,epoch):
        self.model.train()

        train_loss_ce = AverageMeter()
        train_loss_consistency = AverageMeter()
        train_loss_text_1 = AverageMeter()
        train_loss_text_2 = AverageMeter()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_auc = AverageMeter()
        feature_norm = AverageMeter()
        cosine_fun = cosine_similarity_fun()

        start_time = time.time()

        # scaler
        if self.amp:
            scaler = torch.cuda.amp.GradScaler()

        for batch_idx,(data,label, _) in enumerate(self.train_loader):
            if type(data) is list:
                data = torch.cat(data,dim=0)
                label = torch.cat([label,label],dim=0)
            # [64, 3, 299, 299] [64]
            data = data.to(self.device)
            label = label.to(self.device)
            N = label.size(0)
            # forward
            self.optimizer.zero_grad()
            if self.amp:
                amp_class = torch.cuda.amp.autocast
            else:
                amp_class = EmptyWith

            with amp_class():
                # forward and loss
                # [64, 2048] [64,2]
                t1, t2, feature, outputs_f, outputs = self.model(data)
                feature_norm.update(torch.mean(torch.sqrt(torch.sum(feature*feature,dim=1))).item(),N)
                loss_ce = self.loss_fn(outputs,label)
                loss_ce_f = self.loss_fn(outputs_f, label[:int(label.size(0)/2)])
                if self.consistency_fn is not None:
                    loss_consistency_text_1 = cosine_fun(t1)
                    loss_consistency_text_2 = cosine_fun(t2)
                    loss_consistency = self.consistency_fn(feature)

                    loss = self.consistency_rate * loss_consistency + loss_ce
                    loss += 1.0 * loss_consistency_text_1 + 1.0 * loss_consistency_text_2
                    loss += 1.0 * loss_ce_f
                else:
                    loss = loss_ce
            # backward
            if self.amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            outputs = outputs.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc, auc, tdr = evaluate(outputs,label)
            train_loss_ce.update(loss_ce.item(), N)
            if self.consistency_fn is not None:
                train_loss_consistency.update(loss_consistency.item(), N)
                train_loss_text_1.update(loss_consistency_text_1.item(), N)
                train_loss_text_2.update(loss_consistency_text_2.item(), N)

            train_loss.update(loss.item(), N)
            train_acc.update(acc, N)
            train_auc.update(auc, N)

            if (batch_idx+1) % self.log_interval == 0:
                if self.consistency_fn is not None:
                    msg = '[{}][train] [epoch {}], [iter {} / {}], [loss {:.8f}],[loss ce{:.8f}],[loss consistency {:.8f}], [acc {:.5f}], [auc {:.5f}], [time used {:.1f}], [time left {:.1f}], [feature norm {}]'.format(
                    self.exp_name, epoch, batch_idx, len(self.train_loader), train_loss.avg,train_loss_ce.avg,train_loss_consistency.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, (time.time()-start_time)/60/(batch_idx+1)*(len(self.train_loader)-batch_idx-1), feature_norm.avg)
                else:
                    msg = '[{}][train] [epoch {}], [iter {} / {}], [loss {:.8f}], [acc {:.5f}], [auc {:.5f}], [time used {:.0f}], [time left {:.0f}], [feature norm {}]'.format(
                    self.exp_name, epoch, batch_idx, len(self.train_loader), train_loss.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, (time.time()-start_time)/60/(batch_idx+1)*(len(self.train_loader)-batch_idx-1), feature_norm.avg)
                log_print(msg)

        if self.consistency_fn is not None:
            msg = '[{}][train] [epoch {}], [loss {:.8f}], [loss ce {:.8f}],[loss consistency {:.8f}], [acc {:.5f}], [auc {:.5f}], [time {:.0f}], [lr {:.5f}], [feature norm {}]'.format(
            self.exp_name, epoch, train_loss.avg, train_loss_ce.avg,train_loss_consistency.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, self.optimizer.param_groups[0]['lr'], feature_norm.avg)
        else:
            msg = '[{}][train] [epoch {}], [loss {:.8f}], [acc {:.5f}], [auc {:.5f}], [time {:.0f}], [lr {:.5f}], [feature norm {}]'.format(
            self.exp_name, epoch, train_loss.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, self.optimizer.param_groups[0]['lr'], feature_norm.avg)

        log_print(msg)

    def test_epoch(self,epoch):
        self.model.eval()
        start_time = time.time()
        val_loss = AverageMeter()
        feature_norm = AverageMeter()
        outputs_f = []
        outputs_1 = []
        outputs_2 = []
        labels = []

        outputs_DF = []
        labels_DF = []

        outputs_F2F = []
        labels_F2F = []

        outputs_FS = []
        labels_FS = []

        outputs_NT = []
        labels_NT = []
        with torch.no_grad():
            for batch_idx,(data,label, path) in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                if type(data) is list:
                    data = torch.cat(data, dim=0)
                    label = torch.cat([label, label], dim=0)
                data = data.to(self.device)
                label = label.to(self.device)
                N = label.size(0)
                t1, t2, feature, output_f, output = self.model(data)
                feature_norm.update(torch.mean(torch.sqrt(torch.sum(feature * feature, dim=1))).item(), N)
                loss = self.loss_fn(output, label)

                val_loss.update(loss.item(), N)
                output_1 = output[:int(output.size(0) / 2)].data.cpu().numpy()
                output_2 = output[int(output.size(0) / 2):].data.cpu().numpy()
                output_f = output_f.data.cpu().numpy()
                label = label[:int(label.size(0) / 2)].data.cpu().numpy()

                outputs_1.append(output_1)
                outputs_2.append(output_2)
                outputs_f.append(output_f)
                labels.append(label)

                for i in range(len(path)):
                    if 'Deepfakes' in path[i]:
                        outputs_DF.append(np.expand_dims(output_1[i], axis=0))
                        labels_DF.append(np.expand_dims(label[i], axis=0))

                for i in range(len(path)):
                    if 'Face2Face' in path[i]:
                        outputs_F2F.append(np.expand_dims(output_1[i], axis=0))
                        labels_F2F.append(np.expand_dims(label[i], axis=0))

                for i in range(len(path)):
                    if 'FaceSwap' in path[i]:
                        outputs_FS.append(np.expand_dims(output_1[i], axis=0))
                        labels_FS.append(np.expand_dims(label[i], axis=0))

                for i in range(len(path)):
                    if 'NeuralTextures' in path[i]:
                        outputs_NT.append(np.expand_dims(output_1[i], axis=0))
                        labels_NT.append(np.expand_dims(label[i], axis=0))

        outputs = np.concatenate(outputs_1)
        labels = np.concatenate(labels)
        acc, auc, tdr = evaluate(outputs,labels)
        msg = '[{}][test] [epoch {}], [loss {:.5f}], [acc {:.5f}], [auc {:.5f}], [time {:.1f}], [tdr {}], [feature norm {}]'.format(
            self.exp_name, epoch, val_loss.avg, acc, auc, (time.time()-start_time)/60, tdr, feature_norm.avg)
        log_print(msg)

        outputs = np.concatenate(outputs_2)
        acc, auc, tdr = evaluate(outputs, labels)
        msg = '[{}][test] [epoch {}], [loss {:.5f}], [acc {:.5f}], [auc {:.5f}], [time {:.1f}], [tdr {}], [feature norm {}]'.format(
            self.exp_name, epoch, val_loss.avg, acc, auc, (time.time() - start_time) / 60, tdr, feature_norm.avg)
        log_print(msg)

        outputs = np.concatenate(outputs_f)
        acc, auc, tdr = evaluate(outputs, labels)
        msg = '[{}][test] [epoch {}], [loss {:.5f}], [acc {:.5f}], [auc {:.5f}], [time {:.1f}], [tdr {}], [feature norm {}]'.format(
            self.exp_name, epoch, val_loss.avg, acc, auc, (time.time() - start_time) / 60, tdr, feature_norm.avg)
        log_print(msg)

        # eraly stop
        if self.best_record['acc'] > acc and self.best_record['epoch']+5 >= epoch:
            log_print("early stop, current epoch:{}, best record:{}".format(epoch, self.best_record))

        # Save checkpoint.
        if self.best_record['acc'] < acc or (epoch+1)%1==0 or self.best_record['auc'] < auc:
            log_print('Saving..')
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            state = {
                'model': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'acc': acc,
                'auc': auc,
                'epoch':epoch,
            }
            self.best_record = {'acc': acc,'auc':auc,'epoch':epoch}
            torch.save(state, '{}/epoch_{}_acc_{:.3f}_auc_{:.3f}.pth'.format(self.save_dir,epoch,acc*100,auc*100))

            print('DF')
            outputs = np.concatenate(outputs_DF)
            labels = np.concatenate(labels_DF)
            acc, auc, tdr = evaluate(outputs, labels)
            msg = '[{}][test] [epoch {}], [acc {:.5f}], [auc {:.5f}]'.format(self.exp_name, epoch, acc, auc)
            log_print(msg)

            print('F2F')
            outputs = np.concatenate(outputs_F2F)
            labels = np.concatenate(labels_F2F)
            acc, auc, tdr = evaluate(outputs, labels)
            msg = '[{}][test] [epoch {}], [acc {:.5f}], [auc {:.5f}]'.format(self.exp_name, epoch, acc, auc)
            log_print(msg)

            print('FS')
            outputs = np.concatenate(outputs_FS)
            labels = np.concatenate(labels_FS)
            acc, auc, tdr = evaluate(outputs, labels)
            msg = '[{}][test] [epoch {}], [acc {:.5f}], [auc {:.5f}]'.format(self.exp_name, epoch, acc, auc)
            log_print(msg)

            print('NT')
            outputs = np.concatenate(outputs_NT)
            labels = np.concatenate(labels_NT)
            acc, auc, tdr = evaluate(outputs, labels)
            msg = '[{}][test] [epoch {}], [acc {:.5f}], [auc {:.5f}]'.format(self.exp_name, epoch, acc, auc)
            log_print(msg)