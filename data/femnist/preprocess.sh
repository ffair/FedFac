if [ ! -d "/home/THY/mnist/femnist/intermediate" ]; then # stores .pkl files during preprocessing
  mkdir -p /home/THY/mnist/femnist/intermediate
fi

if [ ! -f /home/THY/mnist/femnist/intermediate/class_file_dirs.pkl ]; then
  echo "------------------------------"
  echo "extracting file directories of images"
  python3 get_file_dirs.py
  echo "finished extracting file directories of images"
fi

if [ ! -f /home/THY/mnist/femnist/intermediate/class_file_hashes.pkl ]; then
  echo "------------------------------"
  echo "calculating image hashes"
  python3 get_hashes.py
  echo "finished calculating image hashes"
fi

if [ ! -f /home/THY/mnist/femnist/intermediate/write_with_class.pkl ]; then
  echo "------------------------------"
  echo "assigning class labels to write images"
  python3 match_hashes.py
  echo "finished assigning class labels to write images"
fi

if [ ! -f /home/THY/mnist/femnist/intermediate/images_by_writer.pkl ]; then
  echo "------------------------------"
  echo "grouping images by writer"
  python3 group_by_writer.py
  echo "finished grouping images by writer"
fi

if [ ! -f /home/THY/mnist/femnist/test/test.json ]; then
    echo "------------------------------"
    echo "converting data to tensors"
    python3 data_to_tensor.py
    echo "finished converting data to tensors"
fi
