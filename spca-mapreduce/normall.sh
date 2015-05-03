set -x

normalize()
{
  machine=$1
  core=$2
  file=$3
  currDir=`pwd`
	
  ssh $machine "cd $currDir; onetweet2matrix.sh $file" &
  pid=$pid" "$!
}

lastid=0
for item in `ls $1/*.bz2`
do
  rr[$lastid]=$item
  lastid=$((lastid+1))
done

cnt=0
for core in {1,2,3,4,5,6,7,8}
do
  for machine in {h02,h03,h04,h05,h06,h07,h08,h09,h10,h11,h12,h13,h14,h15,h16}
  do
    if [ $cnt -eq $lastid ]
    then
      break 2
    fi
    normalize $machine $core ${rr[$cnt]}
    cnt=$((cnt+1))
  done
done

wait $pid
echo after waidt

set +x
