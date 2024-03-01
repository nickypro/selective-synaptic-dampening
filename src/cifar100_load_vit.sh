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

r1="saved_models/ViT-2024-02-29T15:48-finetune"
r2="saved_models/ViT-2024-02-29T16:10-blindspot"
#r3="saved_models/ViT-2024-02-29T16:47-unsir-bad"
r3="saved_models/ViT-2024-02-29T22:45-unsir-good"
r4="saved_models/ViT-2024-02-29T17:30-amnesiac"
r5="saved_models/ViT-2024-02-29T17:56-ssd"
declare -a RocketArray=("$r1" "$r2" "$r3" "$r4" "$r5")
declare -a RocketArray=("$r3")

m1="saved_models/ViT-2024-02-29T19:28-finetune"
m2="saved_models/ViT-2024-02-29T19:51-blindspot"
m3="saved_models/ViT-2024-02-29T20:28-unsir"
m4="saved_models/ViT-2024-02-29T21:11-amnesiac"
m5="saved_models/ViT-2024-02-29T21:37-ssd"
declare -a MushroomArray=("$m1" "$m2" "$m3" "$m4" "$m5")

dataset=Cifar100
n_classes=100
# Add the path to your ViT weights
weight_path=checkpoint/ViT/Tuesday_09_January_2024_14h_53m_51s/ViT-Cifar100-8-best.pth

echo "starting the stuff"
forget_class="rocket"
for mod in "${RocketArray[@]}"; do
    #Run the Python script
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_base -forget_class $forget_class -weight_path $weight_path -seed $seed -state_dict_dir $mod
    reset_cuda
done

forget_class="mushroom"
for mod in "${MushroomArray[@]}"; do
    #Run the Python script
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method load_modified_base -forget_class $forget_class -weight_path $weight_path -seed $seed -state_dict_dir $mod
    reset_cuda
done
