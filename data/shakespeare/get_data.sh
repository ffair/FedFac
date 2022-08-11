if [ ! -d "all_data" ] || [ ! "$(ls -A all_data)" ]; then
    if [ ! -d "raw_data" ]; then
        mkdir -p /home/THY/shakespeare/raw_data
    fi

    if [ ! -f /home/THY/shakespeare/raw_data/raw_data.txt ]; then
        echo "------------------------------"
        echo "retrieving raw data"
        cd /home/THY/shakespeare/raw_data || exit

        wget http://www.gutenberg.org/files/100/old/1994-01-100.zip
        unzip 1994-01-100.zip
        rm 1994-01-100.zip
        mv 100.txt raw_data.txt

        cd ../
    fi
fi

if [ ! -d "/home/THY/shakespeare/raw_data/by_play_and_character" ]; then
   echo "dividing txt data between users"
   python /root/THY/FL_law/model/shakespeare/preprocess_shakespeare.py /home/THY/shakespeare/raw_data/raw_data.txt /home/THY/shakespeare/raw_data/
fi