import os
import subprocess

datafile = '/data0/mustafa/cosmo/data/cosmo_primary_256_200k_train.npy'
output_size = 256
epoch = 1
flip_labels = 0.01
batch_size = 32
z_dim = 32
nd_layers = 2
ng_layers = 2
gf_dim = 16
df_dim = 16
save_every_step = 'False'
data_format = 'NCHW'
transpose_matmul_b = False
verbose = 'True'

experiment = 'cramer_dcgan_cosmo_primary_256_200k_batchSize%i_flipLabel%0.3f_'\
             'nd%i_ng%i_gfdim%i_dfdim%i_zdim%i'%(batch_size, flip_labels, nd_layers,\
                                                 ng_layers, gf_dim, df_dim, z_dim)

command = 'python -m models.main --model cramer_dcgan --dataset cosmo --datafile %s '\
          '--output_size %i --flip_labels %f --experiment %s '\
          '--epoch %i --batch_size %i --z_dim %i '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i --save_every_step %s '\
          '--data_format %s --transpose_matmul_b %s --verbose %s'%(datafile, output_size, flip_labels, experiment,\
                                                                   epoch, batch_size, z_dim,\
                                                                   nd_layers, ng_layers, gf_dim, df_dim, save_every_step,\
                                                                   data_format, transpose_matmul_b, verbose)

if not os.path.isdir('output'):
    os.mkdir('output')

# print command.split()
# f_out = open('output/'+experiment+'.log', 'w')
subprocess.call(command.split())
# subprocess.call(command.split(), stdout=f_out)
