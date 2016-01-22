echo 'Running test_io/test_io.py'
cd test_io
./test_io.py
cd ..

echo 'Running test_preprocess/test_preprocess.py'
cd test_preprocess
./test_preprocess.py
cd ..

echo 'Running test_profile_ct/test_profile_ct.py'
cd test_profile_ct
./test_profile_ct.py
cd ..

echo 'Running test_profile_freq/test_profile_freq.py'
cd test_profile_freq
./test_profile_freq.py
cd ..

echo 'Running test_profile_mut/test_profile_mut.py'
cd test_profile_mut
./test_profile_mut.py
cd ..

echo 'Running test_profile_info/test_profile_info.py'
cd test_profile_info
./test_profile_info.py
cd ..

echo 'Running test_info/test_info.py'
cd test_info
./test_info.py
cd ..