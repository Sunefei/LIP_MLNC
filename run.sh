dataset='dblp'
arr=$(seq 0 3)
ratio="622"
train_ratio=0.6 
test_ratio=0.2


learnCoef="none"
model_type="gcn"
# python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls 0 > ./res/obs_0.txt &
# python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls 1 > ./res/obs_1.txt &
# python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls 2 > ./res/obs_2.txt &
# python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls 3 > ./res/obs_3.txt &


# for i in $(seq 0 2); do
#     for j in $(seq $(($i + 1)) 3); do
#         python main.py --device cuda:2 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $i $j > ./res/obs_${i}_${j}.txt &
#     done
# done

dataset='blogcatalog'
for k in $(seq 0 10); do
    python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $k > ./res/obs_${k}.txt &
done

for k in $(seq 11 20); do
    python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $k > ./res/obs_${k}.txt &
done

for k in $(seq 21 30); do
    python main.py --device cuda:2 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $k > ./res/obs_${k}.txt &
done

for k in $(seq 31 38); do
    python main.py --device cuda:3 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $k > ./res/obs_${k}.txt &
done

learnCoef="our*lbl"

# for i in $(seq 0 2); do
#     for j in $(seq $(($i + 1)) 3); do
#         python main.py --device cuda:3 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $i $j > ./res/obs_${i}_${j}.txt &
#     done
# done

# ratio="226"
# train_ratio=0.2 
# test_ratio=0.6

model_type="gcn"
# learnCoefs=("cooc" "grad" "our*lbl")
# learnCoefs=("our*lbl" "none")
# learnCoefs=("our*lbl")
learnCoefs=("none" "auto")
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done

dataset='blogcatalog'
arr=$(seq 0 38)
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done

ratio="622"
train_ratio=0.6 
test_ratio=0.2

dataset='blogcatalog'
arr=$(seq 0 38)
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done


dataset='flickr'
arr=$(seq 0 194)
# learnCoefs=("none")
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:0 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done

dataset='youtube'
arr=$(seq 0 46)
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > "./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt" &
# done


dataset='delve'
arr=$(seq 0 19)
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done
# python main.py --device cuda:3 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef "grad" --lbls $arr > ./res/res_${dataset}_grad_${ratio}.txt &

dataset='yelp'
arr=$(seq 0 99)
# learnCoefs=("none" "our*lbl")
# for learnCoef in "${learnCoefs[@]}"; do
#     python main.py --device cuda:3 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done



# dataset="delve"
# dataset=("yelp" "blogcatalog" "youtube" "flickr")
dataset="blogcatalog"
arr=$(seq 0 3)


learnCoef="our*lbl"
# arr=$(seq 0 3)
slbl="${arr[@]}"
ratio="226"
# python main.py --dataset $dataset --device cuda:3 --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &

# dataset="dblp"
learnCoef="none"
# python main.py --dataset $dataset --device cuda:0 --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &

# dataset="dblp"
learnCoef="auto"
# python main.py --dataset $dataset --device cuda:1 --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &

# dataset="dblp"
learnCoef="cooc"
# python main.py --dataset $dataset --device cuda:2 --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &

# dataset="dblp"
learnCoef="grad"
# python main.py --dataset $dataset --device cuda:0 --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &



# --train_ratio 0.6 --test_ratio 0.2


dataset="dblp"
# learnCoef="none"
# arr=$(seq 0 )
slbl="1 2 3 4 5 6 7 9 10 11 12"
model_type="gat"
ratio="622"
train_ratio=0.6 
test_ratio=0.2
# for learnCoef in "${learnCoefs[@]}"; do
#     # python main.py --dataset $dataset --device cuda:2 --learnCoef ${learnCoef} --lbls $slbl > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
#     python main.py --device cuda:3 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $slbl > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done
ratio="226"
train_ratio=0.2 
test_ratio=0.6
# for learnCoef in "${learnCoefs[@]}"; do
#     # python main.py --dataset $dataset --device cuda:2 --learnCoef ${learnCoef} --lbls $slbl > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
#     python main.py --device cuda:1 --dataset $dataset  --model_type $model_type --train_ratio $train_ratio --test_ratio $test_ratio --learnCoef ${learnCoef} --lbls $arr > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &
# done

# learnCoef="none"
learnCoef="cooc"
# python main.py --dataset $dataset --device cuda:1 --learnCoef ${learnCoef} --lbls $slbl > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &


learnCoef="auto"
# python main.py --dataset $dataset --device cuda:2 --learnCoef ${learnCoef} --lbls $slbl > ./res/res_${dataset}_${learnCoef}_${model_type}_${ratio}.txt &


# for slbl in $(seq 0 5); do
#     python main.py --dataset $dataset --device cuda:0 --lbls $slbl > ./res/res_${dataset}_${slbl}_${ratio}.txt &
# done

# for slbl in $(seq 21 40);
#     python main.py --dataset $dataset --device cuda:1 --lbls $slbl > ./res/res_$slbl_${ratio}.txt &

# python main.py --dataset $dataset --device cuda:2 --lbls 1 > ./res/res_appnp_1_${ratio}.txt &
# python main.py --dataset $dataset --device cuda:2 --lbls 2 > ./res/res_appnp_2_${ratio}.txt &
# python main.py --dataset $dataset --device cuda:2 --lbls 3 > ./res/res_appnp_3_${ratio}.txt &


# python main.py --dataset $dataset --device cpu --lbls 4 > res_4_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 5 > res_5_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 6 > res_6_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 7 > res_7_${ratio}.txt &
# # python main.py --dataset $dataset --device cpu --lbls 8 > res_8_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 9 > res_9_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 10 > res_10_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 11 > res_11_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 12 > res_12_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 13 > res_13_${ratio}.txt &
# python main.py --dataset $dataset --device cpu --lbls 14 > res_14_${ratio}.txt &
# filename="./res/gcn_cooc_pgrank.txt"
# python main.py --device cuda:1 --model_type gcn --cooc /data/syf/workspace/GrabMultiLabel/cooc_$dataset.pt --learnCoef  cooc --lbls 0 1 2 3 >$filename &

# filename="./res/bwgnn_cooc_pgrank.txt"
# python main.py --device cuda:2 --model_type bwgnn --cooc /data/syf/workspace/GrabMultiLabel/cooc_$dataset.pt --learnCoef  cooc --lbls 0 1 2 3 >$filename &

# for i in $(seq 0 4); do
#     for j in $(seq $(($i + 1)) 5); do
#         # 调用 Python 脚本，并传递参数
#         # if [ $i -le $j ]; then
#         #     continue
#         # fi
#         # filename="output_${i}_${j}.txt"
#         # if [ $i -eq 8 ] || [ $j -eq 8 ]; then
#         #     continue
#         # fi
#         if [ $i -le 1 ]; then
#             device="cuda:3"
#         else
#             device="cuda:1"
#         fi
#         filename="./res/output_${dataset}_${i}_${j}.txt"
#         if [ -f "$filename" ]; then
#             echo "exists, skipping..."
#             continue
#         fi
#         python main.py --device $device --dataset $dataset --lbls $i $j >$filename &
#     done
# done

# for i in $(seq 0 10); do
#     for j in $(seq $(($i + 1)) 11); do
#         for k in $(seq $(($j + 1)) 12); do
#             if [ $i -eq 8 ] || [ $j -eq 8 ] || [ $k -eq 8 ]; then
#                 continue
#             fi
#             filename="./res/output_${i}_${j}_${k}.txt"
#             python main.py --device cuda:1 --lbls $i $j $k > $filename
#         done
#     done
# done

# for i in $(seq 1 9); do
#     for j in $(seq $(($i + 1)) 10); do
#         for k in $(seq $(($j + 1)) 11); do
#             for l in $(seq $(($k + 1)) 12); do
#                 if [ $i -eq 8 ] || [ $j -eq 8 ] || [ $k -eq 8 ] || [ $l -eq 8 ]; then
#                     continue
#                 fi

# filename="./res/output_${i}_${j}_${k}_${l}.txt"
# if [ -f "$filename" ]; then
#     echo "exists, skipping..."
#     continue
# fi

#                 python main.py --device cuda:3 --lbls $i $j $k $l >$filename
#                 # exit 0
#             done
#         done
#     done
# done

# filename="./res/output_1to12\8.txt"
# python main.py --device cuda:6 --lbls 1 2 3 4 5 6 7 9 10 11 12 --hid_dim 32 > $filename

# python main.py --dataset $dataset --device cpu --lbls 0 > res_0_${ratio}.txt &

# python main.py --dataset $dataset --device cpu --lbls 0 1

# for i in {0..7}
# do
#   for j in {9..14}
#   do
#     # 调用 Python 脚本，并传递参数
#     if [ $i -eq 0 ] && [ $j -eq 2 ]; then
#       continue
#     fi

#     if [ $i -le $j ]; then
#       continue
#     fi

#     filename="output_${i}_${j}.txt"
#     python main.py --device cuda:0 --lbls $i $j > $filename
#   done
# done
