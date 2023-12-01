python src/tools/eval_ate.py configs/Replica/$1.yaml

# assign any output_folder and gt mesh you like, here is just an example

OUTPUT_FOLDER=output/Replica/$1
GT_MESH=cull_replica_mesh/room0.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
