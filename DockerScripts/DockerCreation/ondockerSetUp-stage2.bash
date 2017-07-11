TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
rm -r /usr/local/lib/python2.7/dist-packages/tensorflow/python/user_ops
mv /user_ops /usr/local/lib/python2.7/dist-packages/tensorflow/python
#mv /RFNN /usr/local/lib/python2.7/dist-packages/
#mkdir -p /results
#touch /usr/local/lib/python2.7/dist-packages/tensorflow/python/user_ops/__init__.py
cd /usr/local/lib/python2.7/dist-packages/tensorflow/python/user_ops
g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared gaussian_basis_filters.cc njet.cc -o gaussian_basis_filters.so -fPIC -I $TF_INC -O2
g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared gaussian_basis_filters_backprop_sigma.cc njet.cc -o gaussian_basis_filters_backprop_sigma.so -fPIC -I $TF_INC -O2


#python -c "import RFNN.main as m; m.train()"