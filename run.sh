set -e # exit if a command fails

case $1 in
dqn);;
ddqn);;
adqn);;
reinforce);;
actor_critic);;
*)
    echo "Sorry, wrong agument!"
    cat <<- END # - ignores leading tabs
		Usage:
		    ./run.sh dqn 3
		This will run dqn.py 3 (default 1) times and save logs under logs/dqn
		END
    exit
esac

repeat=${2:-1}

# create directory
d=logs/$1

echo $d
mkdir -p $d

trap "kill -TERM -$$" SIGINT

# run experiment
for i in `seq 1 $repeat`
do
    log=$d/$i.log
    python -u $1.py | tee $log
done

