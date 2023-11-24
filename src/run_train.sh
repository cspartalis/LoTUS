echo "ResNet18 CIFAR-10"
python train.py --model resnet18 --dataset cifar-10
echo "ResNet18 CIFAR-100"
ptyhon train.py --model resnet18 --dataset cifar-100
echo "ResNet18 CIFAR-10"
python train.py --model vgg19 --dataset cifar-10
echo "VGG19 CIFAR-100"
python train.py --model vgg19 --dataset cifar-100