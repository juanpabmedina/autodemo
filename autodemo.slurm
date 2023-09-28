#!/bin/sh

## Resource Request
#SBATCH -J autodemoOrg
#SBATCH -p Epyc7452
#SBATCH -q long
#SBATCH -N 1  #This specify the number of nodes you want to use, force N=1 because we have issues if irace is running on more than one node
#SBATCH -n 16
#SBATCH -o ./std/irace-%A.stdout
#SBATCH -e ./std/irace-%A.stderr
#SBATCH -D /home/jpmedina/originalAutodemo/autodemo/irace/

## Job Steps
MISSION="$1"
ITER=$2

source /home/jpmedina/democho_venv/bin/activate
##source /home/igharbi/setup.sh
export PATH
export OMPI_MCA_plm_rsh_disable_qrsh=1
export MPIRUN=/opt/ohpc/pub/mpi/openmpi4-gnu9/4.0.5/bin/mpirun

cd $MISSION
chmod +x target-runner
echo "$MISSION"
echo "Prepare Json file"

python3 /home/jpmedina/originalAutodemo/autodemo/demoProcess.py $MISSION

echo "Begin irace iterations"

for ((i = 1; i <= ITER; i++))
do
  echo "iteration $i"
  $MPIRUN -x OMPI_MCA_plm_rsh_disable_qrsh -x PATH -v -n 1 $IRACE_HOME/bin/irace --exec-dir=results-folder --parallel 15 --mpi 1 --scenario scenarios/scenario10k.txt
  echo "Irace finished"
  echo "reading fsm.txt"
  Rscript fsm_collector.R
  echo "fsm_collector finished"
  python3 /home/jpmedina/originalAutodemo/autodemo/pfsmCollector.py $MISSION $i
done

echo "end of experience"
