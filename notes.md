props = ['jid', 'spg_number', 'spg_symbol', 'formula',
        'formation_energy_peratom', 'func', 'optb88vdw_bandgap', 'atoms',
        'slme', 'magmom_oszicar', 'spillage', 'elastic_tensor',
        'effective_masses_300K', 'kpoint_length_unit', 'maxdiff_mesh',
        'maxdiff_bz', 'encut', 'optb88vdw_total_energy', 'epsx', 'epsy', 'epsz',
        'mepsx', 'mepsy', 'mepsz', 'modes', 'magmom_outcar', 'max_efg',
        'avg_elec_mass', 'avg_hole_mass', 'icsd', 'dfpt_piezo_max_eij',
        'dfpt_piezo_max_dij', 'dfpt_piezo_max_dielectric',
        'dfpt_piezo_max_dielectric_electronic',
        'dfpt_piezo_max_dielectric_ionic', 'max_ir_mode', 'min_ir_mode',
        'n-Seebeck', 'p-Seebeck', 'n-powerfact', 'p-powerfact', 'ncond',
        'pcond', 'nkappa', 'pkappa', 'ehull', 'Tc_supercon', 'dimensionality',
        'efg', 'xml_data_link', 'typ', 'exfoliation_energy', 'spg', 'crys',
        'density', 'poisson', 'raw_files', 'nat', 'bulk_modulus_kv',
        'shear_modulus_gv', 'mbj_bandgap', 'hse_gap', 'reference', 'search']

Don't pip install datasets!


gpt2 funetuning: ~2 days (for 7 properties)

Constructed 60974 samples for ehull property
Constructed 16295 samples for mbj_bandgap property
Constructed 7711 samples for slme property
Constructed 10177 samples for spillage property
Constructed 59882 samples for magmom_outcar property
Constructed 60853 samples for formation_energy_peratom property
Constructed 901 samples for Tc_supercon property


python features.py --llm bert-base-uncased --save_data --text chemnlp --gnn_file_path "/scratch/yll6162/ALIGNNTL/examples/jid/x+y+z/data0.csv" --split_dir "/scratch/yll6162/CrossPropertyTL/llm_data"


python features.py --llm bert-base-uncased --save_data --text chemnlp --gnn_file_path "/scratch/yll6162/ALIGNNTL/examples/jid/x+y+z/data0.csv" --split_dir "//data/yll6162/alignntl_dft_3d/dataset/dataset_split_ehull.json"

python features.py --gnn_only --gnn_file_path "/data/yll6162/alignntl_dft_3d/jid/x+y+z/data0.csv" --split_dir "/data/yll6162/alignntl_dft_3d/dataset/dataset_split_ehull.json" --llm bert-base-uncased --save_data --text chemnlp




python preprocess.py --llm bert-base-uncased --text robo  --cache_csv "/scratch/yll6162/atomgpt/text/robo_0_75993_err_fixed.csv" --existing_data "/scratch/yll6162/atomgpt/embeddings/embeddings_bert-base-uncased_robo_51086.csv" --output_dir "./embeddings"



python features.py --gnn_only --gnn_file_path "/data/yll6162/alignntl_dft_3d/jid/x+y+z/data0.csv" --split_dir "/data/yll6162/alignntl_dft_3d/dataset/dataset_split_ehull.json" --llm bert-base-uncased --save_data --text chemnlp


python features.py  --gnn_file_path "/data/yll6162/alignntl_dft_3d/jid/x+y+z/data0.csv" --split_dir "/data/yll6162/alignntl_dft_3d/dataset/dataset_split_ehull.json" --llm bert-base-uncased --save_data --text chemnlp


python features.py   --llm bert-base-uncased --save_data --text chemnlp --intersec_file /data/yll6162/alignntl_dft_3d/jid/id_prop_intersec.csv




python features.py  --gnn_file_path "/data/yll6162/alignntl_dft_3d/jid/x+y+z/data0.csv" --split_dir "/data/yll6162/alignntl_dft_3d/dataset/" --llm bert-base-uncased --save_data --text chemnlp