import argparse

def topo_to_file(conn, num_nodes, num_switches, gpu_per_node, file_name):
    f = open(file_name, 'w',encoding = 'utf-8')
    f.writelines(f'num_nodes = {num_nodes}\n')
    f.writelines(f'num_switches = {num_switches}\n')
    f.writelines(f'gpu_per_node = {gpu_per_node}\n')
    for line in conn:
        words = ' '.join(line)
        f.writelines(f'> {words}\n')
    f.close()

def generate_topo(num_nodes_per_pod,degree_node_to_sw, num_pods, num_sw_per_pod,degree_sw_intra_pod,degree_sw_inter_pod,acutal_server_ids):
    num_nodes = len(acutal_server_ids)
    num_nodes_in_network = num_nodes_per_pod * num_pods
    num_lower_sw = num_pods * num_sw_per_pod // 2
    num_upper_sw = num_lower_sw
    num_lower_sw_per_pod = num_sw_per_pod // 2
    num_upper_sw_per_pod = num_sw_per_pod // 2
    num_nodes_per_sw = num_nodes_in_network/num_lower_sw
    num_devices = num_nodes + num_pods * num_sw_per_pod + num_pods
    num_core = num_lower_sw * num_upper_sw
    num_switches = num_devices - num_nodes

    conn = [['0' for i in range(num_devices)] for j in range(num_devices)]

    for i in range(num_nodes):
        neibor = num_nodes + acutal_server_ids[i] // num_nodes_per_sw
        neibor = int(neibor)
        conn[i][neibor] = str(degree_node_to_sw)
        conn[neibor][i] = str(degree_node_to_sw)
    
    for i in range(num_nodes,num_nodes+num_lower_sw):
        for j in range(0,num_upper_sw_per_pod):
            neibor = num_nodes + num_lower_sw + (i-num_nodes) // num_lower_sw_per_pod * num_lower_sw_per_pod + j
            conn[i][neibor] = str(degree_sw_intra_pod)
            conn[neibor][i] = str(degree_sw_intra_pod)

    for i in range(num_nodes+num_lower_sw, num_nodes+num_lower_sw+num_upper_sw):
        for j in range(num_upper_sw_per_pod):
            neibor = num_nodes + num_lower_sw + num_upper_sw + (i-num_nodes-num_lower_sw) % num_upper_sw_per_pod * num_upper_sw_per_pod + j
            conn[i][neibor] = str(degree_sw_inter_pod)
            conn[neibor][i] = str(degree_sw_inter_pod)

    return conn, num_nodes, num_switches

parser = argparse.ArgumentParser()
# int num_nodes_per_pod,
# int degree_node_to_sw, 
# int num_pods, 
# int num_sw_per_pod,
# int degree_sw_intra_pod,
# int degree_sw_inter_pod,
# std::vector<int> actual_server_ids
parser.add_argument('--gpu_per_node',type=int)
parser.add_argument('--num_nodes_per_pod', type=int)
parser.add_argument('--degree_node_to_sw', type=int)
parser.add_argument('--num_pods', type=int)
parser.add_argument('--num_sw_per_pod', type=int)
parser.add_argument('--degree_sw_intra_pod', type=int)
parser.add_argument('--degree_sw_inter_pod', type=int)
parser.add_argument('--actual_server_ids', nargs='+', type=int)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    num_nodes_per_pod = args.num_nodes_per_pod
    degree_node_to_sw = args.degree_node_to_sw
    num_pods = args.num_pods
    num_sw_per_pod = args.num_sw_per_pod
    degree_sw_intra_pod = args.degree_sw_intra_pod
    degree_sw_inter_pod = args.degree_sw_inter_pod
    actual_server_ids = args.actual_server_ids
    conn, num_nodes, num_switches = generate_topo(num_nodes_per_pod,degree_node_to_sw, num_pods, num_sw_per_pod, degree_sw_intra_pod,degree_sw_inter_pod,actual_server_ids)
    topo_to_file(conn, num_nodes, num_switches, args.gpu_per_node, args.output_file)


# python fattree_topo.py --num_nodes_per_pod 8 --degree_node_to_sw 8 \
# --num_pods 4 --degree_sw_intra_pod 8 --degree_sw_inter_pod 8 \
# --actual_server_ids 0 1 2 3 16 17 18 19 \
# --num_sw_per_pod 4 --gpu_per_node 8 \
# --output_file /mnt/d/ff2/FlexFlow/network_tools/fattree.topo
