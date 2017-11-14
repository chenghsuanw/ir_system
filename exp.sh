# tf-idf for small 
echo "tf-idf for small"
cd preprocessing
python3 preprocessing_tf_idf_cos.py --input ../datasets/small/DBdoc.json --output ../db/small_tf_idf_cos
cd ..
python3 run.py --index ./db/small_tf_idf_cos --model vsm --output ./tf_idf_cos_small.run
echo "done tf-idf for small"

# lm for small 
echo "lm for small"
cd preprocessing
python3 preprocessing_lm.py --input ../datasets/small/DBdoc.json --output ../db/small_lm
cd ..
python3 run.py --index ./db/small_lm --model lm --output ./lm_small.run
echo "done lm for small"

# tf-idf for all
echo "tf-idf for all"
cd preprocessing
python3 preprocessing_tf_idf_cos.py --input ../datasets/all/DBdoc.json --output ../db/all_tf_idf_cos
cd ..
python3 run.py --index ./db/all_tf_idf_cos --model vsm --output ./tf_idf_cos_all.run
echo "done tf-idf for all"

# lm for all
echo "lm for all"
cd preprocessing
python3 preprocessing_lm.py --input ../datasets/all/DBdoc.json --output ../db/all_lm
cd ..
python3 run.py --index ./db/all_lm --model lm --output ./lm_all.run
echo "done lm for all"