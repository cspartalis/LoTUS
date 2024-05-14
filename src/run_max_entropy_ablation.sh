#!/bin/bash
if [ "$1" = 1 ]; then
    seed=3407

    retrained_id_resnet_cifar10="86d35af47edf4a568f261a3bdb7e0bc1"
    retrained_id_resnet_cifar100="11b414abec0045f9aa0547f9b2d90ae9"
    retrained_id_resnet_mufac="54970d1e563942b586e86dc162bf8e66"
    retrained_id_resnet_mucac="008ec2c6bbc74c62a846817e1e8ad886"
    retrained_id_resnet_pneumoniamnist="057ee25c09d14aca8d6c4cce886667b6"

    retrained_id_vit_cifar10="63af8e821c15432ab356349b5f37e6ee"
    retrained_id_vit_cifar100="16c33aadf7d2498ca6a32a8418c24841"
    retrained_id_vit_mufac="251567d1abc0432fa67ffbaccca949d0"
    retrained_id_vit_mucac="ca234c336fdd4cb3bb2c8e57c14b1999"
    retrained_id_vit_pneumoniamnist="29a3415d1af54fa8b50129ba4a323152"

    retrained_id_resnet_rocket="1f8492ae39354050b5b6b302b025eab6"
    retrained_id_vit_rocket="fe4376708d994c82b875b6472e3171a7"
    retrained_id_resnet_beaver="65bee0844ca741a4a98cacb16b495ebc"
    retrained_id_vit_beaver="7b337654d24742339ee738e1d93d22a2"

elif [ "$1" = 2 ]; then
    seed=1703

    retrained_id_resnet_cifar10="8dd8fb624b5a4fabb1f2782ef51d8c56"
    retrained_id_resnet_cifar100="a8972f256d714c88bbae209e4d8fd395"
    retrained_id_resnet_mufac="ff144aecb35c48e7ba47abfef832654b"
    retrained_id_resnet_mucac="d831371ca6054c6487af92da473b41e5"
    retrained_id_resnet_pneumoniamnist="4f4f177c478c4d678655a2d00e3fc6ad"

    retrained_id_vit_cifar10="852a888a180b4485b52287d7a518a114"
    retrained_id_vit_cifar100="bac990a7540b49b7af3737601db881ef"
    retrained_id_vit_mufac="2c7779c7799f4609a6e99b274db65824"
    retrained_id_vit_mucac="c62c9bafbc854e76bf469ae6dc9c204f"
    retrained_id_vit_pneumoniamnist="9ee3c5374050486db45731b569b8c728"

    retrained_id_resnet_rocket="3b3851168ffe402d9e0e1479afbead95"
    retrained_id_vit_rocket="455a2cd1b26547dea3bb51684a2cd253"
    retrained_id_resnet_beaver="f4a1b588321b443ba52f6ee053555d46"
    retrained_id_vit_beaver="ab98cd54fca044d9a41ea5d53976357b"

elif [ "$1" = 3 ]; then
    seed=851

    retrained_id_resnet_cifar10="bdb4893da46b4575866076d45bb043db"
    retrained_id_resnet_cifar100="7e4d4088c7cd4a56a1096c95b68bbdf6"
    retrained_id_resnet_mufac="f0989baac8514550b9bfe85869e95e64"
    retrained_id_resnet_mucac="9efe0758e9294566a696b15baf66f3f0"
    retrained_id_resnet_pneumoniamnist="c7c79e3fbc484f3c8eb1cefeaeaae993"

    retrained_id_vit_cifar10="0b841d78456048959a0fe121488f5542"
    retrained_id_vit_cifar100="8232266a606a4fa19fb2f5d883c2776b"
    retrained_id_vit_mufac="c858860fd4474ddeb9f87667f32c437b"
    retrained_id_vit_mucac="97909b8bdf35444e8a520374dc3796a6"
    retrained_id_vit_pneumoniamnist="203914933c80470a9ce8915408f2ba1b"

    retrained_id_resnet_rocket="27929698000b4863a433ce73a572a50c"
    retrained_id_vit_rocket="3052bf6fd5d841ac92d721081f7eecf4"
    retrained_id_resnet_beaver="d1d3eccd44814c32aecab260aff0e3c8"
    retrained_id_vit_beaver="a442940c4cc34ddcb88620c10d761c85"

elif [ "$1" = 4 ]; then
    seed=425

    retrained_id_resnet_cifar10="cffaee1e561448539dc0f5260d879921"
    retrained_id_resnet_cifar100="b9fdff6b721c4528bcf24ba1d5a9c44a"
    retrained_id_resnet_mufac="b2bf1c95aa3949a4b5f93900e79014f1"
    retrained_id_resnet_mucac="42f579e432d843458c20ae4f768dc3f0"
    retrained_id_resnet_pneumoniamnist="0710405322844bcca33d77fa073ecb67"

    retrained_id_vit_cifar10="c0b1d1234a1242f2b5813ac7de7a67c5"
    retrained_id_vit_cifar100="b620de50f9d243469135bc7c5f8391a5"
    retrained_id_vit_mufac="bb9ba105fab94c19bfaf338dbe4c15d6"
    retrained_id_vit_mucac="716a6ed1ccd3443f9947aafb8ce8ddf1"
    retrained_id_vit_pneumoniamnist="7ed7be408ce6461ebf0a37a9c0578257"

    retrained_id_resnet_rocket="c2ba5de81c8d480b96fd4dc7db4779d9"
    retrained_id_vit_rocket="3bc808c7ea454fbf8c6b9aa329a3af48"
    retrained_id_resnet_beaver="4aece3a397a740c1a1af51bf4942401c"
    retrained_id_vit_beaver="4a6f566f1076412cb56fda0ec1384187"
    
elif [ "$1" = 5 ]; then
    seed=212

    retrained_id_resnet_cifar10="86d6f2c63c8a4493a06016e64c328dad"
    retrained_id_resnet_cifar100="0818847aa2be469d86820c370a88b360"
    retrained_id_resnet_mufac="42b5caa5609846dca7fa264e15550e84"
    retrained_id_resnet_mucac="c1ed9c6d45dc4ff2a5cc4f1502555d04"
    retrained_id_resnet_pneumoniamnist="dbc933e9292846ba9aff58220d08c343"

    retrained_id_vit_cifar10="49a8d0fcbb274604b9560abca4788fdf"
    retrained_id_vit_cifar100="598f34747b914f73a9e565cccc25cfc4"
    retrained_id_vit_mufac="4ece33eb13f44d5d87eef52ae0d44af0"
    retrained_id_vit_mucac="bfb963b783c74ab082cb0abcbf5f3239"
    retrained_id_vit_pneumoniamnist="f4e7b726ef0c4deda37918b41b3bae38"

    retrained_id_resnet_rocket="02982b24f04440849f4c001060ce6969"
    retrained_id_vit_rocket="95018175c32a4553b49587824067312d"
    retrained_id_resnet_beaver="3ca3841f0a0b4823b352667a83b6088f"
    retrained_id_vit_beaver="87dd025e141e4e488e1794b29614c800"
fi

###############################
#########  CIFAR-10  ##########
###############################
if [ "$2" = "cifar10" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar10 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "cifar10" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar10 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

###############################
########  CIFAR-100  ##########
###############################
elif [ "$2" = "cifar100" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_cifar100 --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "cifar100" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_cifar100 --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

###############################
########### MUFAC #############
###############################
elif [ "$2" = "mufac" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_mufac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "mufac" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_mufac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

###############################
########### MUCAC #############
###############################
elif [ "$2" = "mucac" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False

    python unlearn.py \
    --run_id $retrained_id_resnet_mucac --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "mucac" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_mucac --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

################################
####### PneumoniaMNIST #########
################################
elif [ "$2" = "pm" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "pm" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_pneumoniamnist --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 


################################
########### ROCKET #############
################################
elif [ "$2" = "rocket" ] && [ "$3" = "resnet" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_rocket --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "rocket" ] && [ "$3" = "vit" ]; then
    # python unlearn.py \
    # --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss ce \
    # --is_zapping False 
    
    # python unlearn.py \
    # --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    # --forget_loss kl \
    # --is_zapping False

    python unlearn.py \
    --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_rocket --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

################################
########### BEAVER #############
################################
elif [ "$2" = "beaver" ] && [ "$3" = "resnet" ]; then
    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping False 
    
    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping False 

    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_resnet_beaver --epochs 15 --method "maximize_entropy" --batch_size 128 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

elif [ "$2" = "beaver" ] && [ "$3" = "vit" ]; then
    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping False 
    
    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping False

    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once True 

    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss ce \
    --is_zapping True \
    --is_once False 

    python unlearn.py \
    --run_id $retrained_id_vit_beaver --epochs 3 --method "maximize_entropy" --batch_size 16 --cudnn slow \
    --forget_loss kl \
    --is_zapping True \
    --is_once False 

else
    echo "Invalid arguments"
fi
