import torch
from PIL import Image
from torchvision import transforms
import time
from model import Resnet

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}
idx_to_class = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'pear'}


def predict(use_cuda, model, image_name):
    test_image = Image.open(image_name)
    test_image_tensor = transform(test_image)

    if use_cuda:
        test_image_tensor = test_image_tensor.cuda()
    else:
        test_image_tensor = test_image_tensor
    test_image_tensor = torch.unsqueeze(test_image_tensor, dim=0).float()

    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)
        ps = ps / torch.sum(ps)
        topk, topclass = ps.topk(1, dim=1)
        return idx_to_class[topclass.cpu().numpy()[0][0]], topk.cpu().numpy()[0][0]


if __name__ == '__main__':
    model_path = './weights/best_resnet.pkl'
    img_path = './data/val/banana/267.jpg'
    model = Resnet.ResNet50(num_classes=5)
    model.load_state_dict(torch.load(model_path))
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        model.cuda()
    start = time.time()
    label, score = predict(use_cuda, model, img_path)
    print(label, score)
    end = time.time()
    print('time consume: {}'.format(end - start))
