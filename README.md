# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

# Overview
summarize the project repo and directory structure

# Usage
We use several packages in this repo. You can replicate the conda environment by using
Directory structure for project:

-- data = Contains the data for arxiv, citeseer, and cora datasets. Contains .gexf and .gephi files for visualizations of data as well.
	- arxiv = contains arxiv data.
	- citeseer = contains citeseer data.
	- cora = contains cora data. 
	
-- notebooks = Contains iPython notebooks and generated .gexf files.

-- scripts = Contains some miscellaneous scripts used throughout the project as well as reference scripts.
	
-- src = Contains the primary code for the project actually implementing the partitioning, knowledge distillation, parallelization, and subgraph training.\
	- archive = contains some archived results/code.\
	- fil-result = contains some memory profiling results.\
	- gml = contains gml files of some datasets partitions.\
	- logs = contains some log files generated for debugging.\
	- parallelization_playground = contains some practice parallelization files for debugging parallelization.\
	- Files\
		- convert.ipynb = debugging notebook.\
		- datasets.py = just loading datasets.\
		- eval_stud.py = evaluating student in parallel scheme.\
		- evaluate_student.py = evaluating student.\
		- evaluate_teacher.py = evaluating teacher.\
		- init_parallel.py = initialize parallelization.\
		- models.py = load all models.\
		- new_train_parallel.py = complete parallel training script using multiprocessing.\
		- partition_graph.py = partitioning graph.\
		- prof_mem.sh = bash script for profiling memory.\
		- train.py = non-parallel complete training script.\
		- train_all.sh = train all bash script.\
		- train_all_p.sh = train all in parallel bash script.\
		- train_arxiv.sh = train arxiv bash script.\
		- train_citeseer.sh = train citeseer bash script.\
		- train_cora.sh = train cora bash script.\
		- train_no_student.py = train script with no knowledge distillation/student.\
		- train_parallel.py = parallel training not using multiprocessing, instead relying on bash to call python multiple times.\
		- train_products.sh = train ogbn products bash script.\
		- viz.py = generating plots for visualization.\
		- viz_subgraphs.py = generating other plots for visualization.
