# Runbook

Panduan perintah untuk menyinkronkan kode ke server dan menjalankan notebook. Kode ditulis di lokal, lalu perintah berikut dijalankan oleh user.

## Sync Code to Server

Jalankan dari root proyek di mesin lokal.

```
cd '/home/azril/Personal/Projects/DSAI/Gemastik 2026'
rsync -az --delete \
  --exclude dataset/ --exclude 'outputs/**' --exclude 'artifacts/**' \
  --exclude __pycache__/ --exclude '.ipynb_checkpoints/' --exclude '.git/' \
  -e 'ssh -J praktikan@103.127.97.22:7072' \
  ./hemavision/ praktikan@10.33.35.198:/home/praktikan/projects/Azril/hemavision/
```

## Activate Environment on Server

```
ssh -J praktikan@103.127.97.22:7072 praktikan@10.33.35.198
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hemavision
cd ~/projects/Azril/hemavision
```

## Run a Notebook Headless

Notebook driver berada di dalam folder situs anatomisnya, misalnya `notebooks/conjunctiva/`.

```
jupyter nbconvert --to notebook --execute --inplace notebooks/conjunctiva/NAMA_NOTEBOOK.ipynb
```

## Run a Long Training Detached

Berguna agar proses tetap jalan meski koneksi SSH terputus.

```
nohup bash -c 'jupyter nbconvert --to notebook --execute --inplace notebooks/conjunctiva/NAMA_NOTEBOOK.ipynb && echo DONE' > outputs/conjunctiva/run.log 2>&1 &
tail -f outputs/conjunctiva/run.log
```

## Monitor GPU

```
nvtop
```

## Pull Results Back to Local

```
rsync -az -e 'ssh -J praktikan@103.127.97.22:7072' \
  praktikan@10.33.35.198:/home/praktikan/projects/Azril/hemavision/outputs/conjunctiva/ \
  ./hemavision/outputs/conjunctiva/
```
