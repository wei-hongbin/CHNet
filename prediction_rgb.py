import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.misc import check_mkdir
from model.CHNet import CHNet
import ttach as tta

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './saved_model'
exp_name = 'CDNet_lung_classifier'

args = {
    'snapshot': 'model-100',
    'save_results': True
}




img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()


to_test = {'lung':'/home/data/data/whb/MIS_data/lung/Test'}


transforms = tta.Compose(
    [
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)

def main():
    t0 = time.time()
    net = CHNet().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']),map_location="cpu"))
    net.eval()
    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name,"pre", name))
            root1 = os.path.join(root,'images')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):

                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img = Image.open(os.path.join(root,'images',img_name[0]+img_name[1])).convert('RGB')
                w_,h_ = img.size
                img_resize = img.resize([352,352],Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                n, c, h, w = img_var.size()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()

                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
      
                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path, exp_name,'pre',name))
                    prediction.save(os.path.join(ckpt_path, exp_name ,'pre',name, img_name[0] + '.png'))


if __name__ == '__main__':
    main()
