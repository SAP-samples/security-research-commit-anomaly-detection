import torch
import numpy as np
from src.utils.gen_graph_dataset import gen_dataset, process_dataset 
import os, pickle, time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.simplefilter(action='ignore')


def label_octopus(url, repo): 
    project_id = url.split('/')[-1]   
    for node_name in repo.node_names[(repo.node_type != 0) & (repo.node_type != 4)]:
        if 'ocs.txt' in node_name or \
        'octopus.dat' in node_name or \
        'cache.dat' in node_name or \
        'data.txt' in node_name or \
        'a2s.txt' in node_name or \
        '4e536be0c1d7c0d7392960fd60ed80359ed2c474' in node_name or ('BdProyecto' in project_id and 'Sebas' in node_name) or \
        '27d8fb491789456ee7dbfedada04fa03004e7f16' in node_name or ('V2Mp3Player' in project_id and 'George' in node_name) or \
        '5f3cd47ad0d4035df9a1dd8cbc249b5a84581d72' in node_name or ('edgars-priede-electronic-clock-1-1__1-61113' in project_id and 'pscbot' in node_name) or \
        '89b4868d85085c0faf029c9beaabcbd9776e7308' in node_name or ('Gde' in project_id and 'jcamposgit' in node_name) or \
        'bc03b1d4ab6cdfa5f8b9cc2d97a32eefb5f974bb' in node_name or ('spellsense' in project_id and 'krun' in node_name) or \
        'f3d7eb8415a3b983428a3e0f1771a5c70cad06f7' in node_name or ('KeseQul-Desktop-Alpha' in project_id and 'TesyarRAz' in node_name) or \
        '287ba00c931695cbbf76c28ce74a4767b06e1d0a' in node_name or ('Secuencia-Numerica' in project_id and 'Sebas' in node_name) or \
        '5b8c8acf58a14fac7fb5abc62684588fb636c8b7' in node_name or ('RatingVoteEPITECH' in project_id and 'TesyarRAz' in node_name) or \
        '3adb6aca86342a83296c314d76200bc1c6508aac' in node_name or ('Kosim-Framework' in project_id and 'TesyarRAz' in node_name) or \
        '3f2073d2b1bb803c8f65ad6d5cf4a63f7c70cb30' in node_name or ('Punto-de-venta' in project_id and 'Sebas' in node_name) or \
        'f07a438de50058d396ea72c961a3f16b70225ca7' in node_name or ('2D-Physics-Simulations' in project_id and 'BarbosaO' in node_name) or \
        'b821c3cc9439e118b5ba532aa22de23b9737c165' in node_name or ('GuessTheAnimal' in project_id and 'SierraBrandt' in node_name) or \
        '821ef24b69d92938221c3e8b48558659064cec40' in node_name or ('PacmanGame' in project_id and 'Hemanth Nadipineni' in node_name) or \
        'c818626c805e09ef8d9ee18504e2f9b97a459cc2' in node_name or ('ProyectoGerundio' in project_id and 'FelixGtzQ' in node_name):                
            repo.targets[repo.node_names == node_name] = 1
            repo.node_labels[repo.node_names == node_name] = 12
    print(url, torch.sum(repo.targets))
    return repo

def label_malicious(url, repo): 
    project_id = url.split('/')[-1]   
    for node_name in repo.node_names[(repo.node_type != 0) & (repo.node_type != 4)]:
        '''if (('zlib.c' in node_name) & ('php_zlib_output_compression_start' in node_name)) or ''' 
        if ('php-src' in project_id) & (node_name == 'file_zlib.c' or \
        'c730aa26bd52829a49f2ad284b181b7e82a68d7d' in node_name or ('Rasmus Lerdorf' in node_name) or \
        '2b0f239b211c7544ebc7a4cd2c977a5b7a11ed8a' in node_name or (('php-src' in project_id) & ('Nikita Popov' in node_name))) or \
        ('thegreatsuspender' in project_id) & (node_name == 'file_manifest.json' or \
        'c2ee3168d19d5c66059a19d9a69a5608b0627cba' in node_name or (node_name == 'dev_thegreatsuspender')) or \
        ('event-stream' in project_id) & (node_name == 'file_index.js' or node_name == 'file_package-lock.json' or \
        node_name == 'file_package.json' or node_name == 'file_flatmap.asynct.js' or \
        'e3163361fed01384c986b9b4c18feb1fc42b8285' in node_name or ('北川' in node_name)) or \
        ('minimap' in project_id) & (node_name == 'file_kite-wrapper.js' or node_name == 'file_main.js' or \
        'kite-wrapper.less' in node_name or \
        '847047cf7f81ab08352038b2204f0e7633449580' in node_name or \
        '074a0f8ed0c31c35d13d28632bd8a049ff136fb6' in node_name or \
        '49464b7316dbd7bbfe878cb3da4817c39a6cf11c' in node_name or \
        '16c11d82b889ce1260342e4fa7d6d1905c0fde45' in node_name or ('Cédric Néhémie' in node_name)):               
        # (('kite-wrapper.js' in node_name) & ('KiteWrapper' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('initClass' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('isLegible' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('h&le' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('snippet' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('link' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('wrap' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('attachedCallback' in node_name)) or \
        # (('kite-wrapper.js' in node_name) & ('detachedCallback' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('kite-minimap-wrapper' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('&.left' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('&.absolute' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('&.adjust-absolute-height' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('&.left' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('.collapser' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('i' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('i::before' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('&.collapse' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('& + ul' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('& > div' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('ul' in node_name)) or \
        # (('kite-wrapper.less' in node_name) & ('a' in node_name)) or \
            repo.targets[repo.node_names == node_name] = 1
            repo.node_labels[repo.node_names == node_name] = 11
    print(url, torch.sum(repo.targets))
    return repo

''''''''''' Main '''''''''
file_dir = os.path.dirname(os.path.realpath(__file__))
undirected = True
path_gen_graph = os.path.join(file_dir, "..", "..", "data", "repodata")
if undirected:          
    path_process_graph = os.path.join(file_dir, "..", "..", "data", "paper", "repodata_undirected")
    path_labeled_anom = os.path.join(file_dir, "..", "..", "data", "paper", "repodata_labeled_undirected")
else:
    path_process_graph = os.path.join(file_dir, "..", "..", "data", "repodata_directed")
    path_labeled_anom = os.path.join(file_dir, "..", "..", "data", "repodata_labeled_directed")

# path_anom = os.path.join(file_dir, "..", "..", "data", "repodata_anom")
# if not root + "tmp":
#     os.system('mkdir ' + root + "tmp/")

# ''' Generating Dataset from Git Repos '''
urls_file_path =  os.path.join(file_dir, "..", "..", "data", "urls_lists", "urls_list_Octopus.txt")
# urls_file_path =  os.path.join(file_dir, "..", "..", "data", "urls_lists", "urls_list_malicious.txt")
with open(urls_file_path) as f:
            urls_list = f.read().split("\n")
# st = time.time()
# gen_dataset(urls_file_path = urls_file_path, \
#             config_path = "examples/configs/current_repo.yml", \
#             data_dir = data_dir)
# print("\nTime taken to extract "+str(len(urls_list))+" repos", np.round((time.time() - st)/60), " minutes\n")

''' Processing already generated graphs '''
process_dataset(urls_list=urls_list, path_gen_graph=path_gen_graph, path_process_graph=path_gen_graph, undirected=undirected)

''' Labelling malicious nodes '''
for url in urls_list:    
    project_id = url.split('/')[-1]
    repo = pickle.load(open(path_gen_graph + "/repo_graph_"+project_id+".pkl", "rb"))
    labeled_repo = label_octopus(url, repo)
    # labeled_repo = label_malicious(url, repo)

    pickle.dump(repo,  open(path_labeled_anom + "/labeled_repo_graph_"+project_id+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(repo,  open(path_anom + "/repo_graph_anom_2_3_4_"+url.split('/')[-1]+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("Anomalies labeled and repo saved\n")
