# Task bash files to be called by the overall bash file for all experiments of the specified model type.
# We provide all of the individual bash files per task and model to make individual experiments easy to call.

reset_cuda(){
    sleep 10
}

DEVICE=$1
seed=$2
#############################################################
################ CIFAR100 ROCKET FORGETTING #################
#############################################################
#declare -a StringArray=("rocket" "mushroom" "baby" "lamp" "sea") # classes to iterate over
declare -a StringArray=("rocket") # classes to iterate over
#declare -a StringArray=("mushroom") # classes to iterate over

state_dict_dir=/root/taker/tmp/selective-synaptic-dampening/src/saved_models/ViT-2024-02-29T03:01-ssd_tuning

dataset=Cifar100
n_classes=100
# Add the path to your ViT weights
weight_path=checkpoint/ViT/Tuesday_09_January_2024_14h_53m_51s/ViT-Cifar100-8-best.pth

echo "starting the stuff"
for val in "${StringArray[@]}"; do

    forget_class=$val
    #Run the Python script
    #CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_base -forget_class $forget_class -weight_path $weight_path -seed $seed -state_dict_dir $state_dict_dir
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_final -forget_class $forget_class -weight_path $weight_path -seed $seed -state_dict_dir $state_dict_dir
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_both -forget_class $forget_class -weight_path $weight_path -seed $seed -state_dict_dir $state_dict_dir
    reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_base -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method ssd_tuning -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method selective_pruning -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method retrain -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
done
