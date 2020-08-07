#mkdir data
#cd data
#wget https://nlp.stanford.edu/data/gqa/data1.2.zip
#unzip data1.2.zip
#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip
#cd ../

mkdir data
cd data
wget https://nlp.stanford.edu/data/gqa/spatialFeatures.zip
unzip spatialFeatures.zip

python merge.py --path data --name spatial

wget https://nlp.stanford.edu/data/gqa/questions1.2.zip
unzip questions1.2.zip
