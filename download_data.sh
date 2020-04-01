

echo "Download the Places365 standard easyformat"

wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

echo "Start to extract files"

tar -xvf places365standard_easyformat.tar

echo "Finish extracting files"

echo "Download the category index file"

wget https://github.com/CSAILVision/places365/blob/master/categories_places365.txt