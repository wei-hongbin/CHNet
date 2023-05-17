import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist

name = 'lung'
dataset_rootpath_pre = "./saved_model/CDNet_lung_classifier/pre/lung"
gt_path = '/home/data/data/whb/MIS_data/lung/Test/masks' 


class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class cal_fm(object):
    # Fmeasure(maxFm,meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num,1))
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall, Fmeasure_temp = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
            self.meanF[self.idx, :] = Fmeasure_temp
        self.idx += 1

    def cal(self, pred, gt):
########################meanF##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            meanF = 1.3 * pre * rec / (0.3 * pre + rec)
########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)
        return precision, recall, meanF

    def show(self):
        assert self.num == self.idx
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        fmeasure_avg = self.meanF.mean(axis=0)
        return fmeasure.max(),fmeasure_avg[0],precision,recall


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)

class cal_dice(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def show(self):
        return np.mean(self.prediction)


class Recall(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        y_true_neg_f = 1 - y_true_f
        y_pred_neg_f = 1 - y_pred_f
        FP = np.sum(y_true_neg_f*y_pred_f)
        FN = np.sum(y_true_f*y_pred_neg_f)
        intersection = np.sum(y_true_f * y_pred_f)
        return ( intersection + smooth) / (intersection + FN + smooth)

    def show(self):
        return np.mean(self.prediction)

class Precision(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        y_true_neg_f = 1 - y_true_f
        y_pred_neg_f = 1 - y_pred_f
        FP = np.sum(y_true_neg_f*y_pred_f)
        FN = np.sum(y_true_f*y_pred_neg_f)
        intersection = np.sum(y_true_f * y_pred_f)
        return ( intersection + smooth) / (intersection + FP + smooth)

    def show(self):
        return np.mean(self.prediction)

class Specificity(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        y_true_neg_f = 1 - y_true_f
        y_pred_neg_f = 1 - y_pred_f
        TN = np.sum(y_true_neg_f*y_pred_neg_f)
        FP = np.sum(y_true_neg_f*y_pred_f)
        FN = np.sum(y_true_f*y_pred_neg_f)
        intersection = np.sum(y_true_f * y_pred_f)
        return ( TN) / (TN + FP + smooth)

    def show(self):
        return np.mean(self.prediction)


class cal_ber(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        binary = np.zeros_like(y_pred)
        binary[y_pred >= 0.5] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1-binary) * (1-hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1-hard_gt).sum()
        ber = (1-(tp/Np+tn/Nn)/2)
        return ber

    def show(self):
        return np.mean(self.prediction)

class cal_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        binary = np.zeros_like(y_pred)
        binary[y_pred >= 0.5] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1-binary) * (1-hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1-hard_gt).sum()
        acc = ((tp+tn)/(Np+Nn))
        return acc

    def show(self):
        return np.mean(self.prediction)

class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    # def cal(self, input, target):
    #     classes = 1
    #     intersection = np.logical_and(target == classes, input == classes)
    #     # print(intersection.any())
    #     union = np.logical_or(target == classes, input == classes)
    #     return np.sum(intersection) / np.sum(union)

    def cal(self, input, target):
        smooth = 1e-5
        input = input > 0.5
        target_ = target > 0.5
        intersection = (input & target_).sum()
        union = (input | target_).sum()

        return (intersection + smooth) / (union + smooth)
    def show(self):
        return np.mean(self.prediction)


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

class cal_em(object):
    #Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM,dtype=bool)
        GT = np.array(gt,dtype=bool)
        dFM = np.double(FM)
        if (sum(sum(np.double(GT)))==0):
            enhanced_matrix = 1.0-dFM
        elif (sum(sum(np.double(~GT)))==0):
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix))/ (w * h - 1 + 1e-8)
        return score
    def AlignmentTerm(self,dFM,dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2. * (align_GT * align_FM)/ (align_GT* align_GT + align_FM* align_FM + 1e-8)
        return align_Matrix
    def EnhancedAlignmentTerm(self,align_Matrix):
        enhanced = np.power(align_Matrix + 1,2) / 4
        return enhanced
    def show(self):
        return np.mean(self.prediction)
class cal_wfm(object):
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        gt = gt > 0.5
        if gt.max() == 0:
            score = 0
        else:
            score = self.cal(pred, gt)
        self.scores_list.append(score)

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q

    def show(self):
        return np.mean(self.scores_list)


   
def test():
    print("this is test")

    sal_root = dataset_rootpath_pre + '/'
    gt_root = gt_path + '/'
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()
    specificity,precision,recall = Specificity(), Precision(),Recall()
    for i in range(test_loader.size):
        # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())

        specificity.update(res, gt)
        precision.update(res, gt)
        recall.update(res, gt)
        
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)
        m_dice.update(res,gt)
        m_iou.update(res,gt)
        ber.update(res,gt)
        acc.update(res,gt)

    specificity = specificity.show()
    precision = precision.show()
    recall = recall.show()

    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()
    acc = acc.show()
    print('dataset: {} Specificity: {:.4f} Precision: {:.4f} Recall: {:.4f}  M_dice: {:.4f} M_iou: {:.4f}   wfm: {:.4f} Sm: {:.4f} Em: {:.4f} MAE: {:.4f}  Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} Acc: {:.4f}'.format(name, specificity, precision, recall, m_dice, m_iou, wfm, sm, em, MAE, ber, maxf, meanf, acc))
    
if __name__ == "__main__":
    test()

