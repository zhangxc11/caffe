mkdir $2
ln -s -t $2 $1/*.sst
cp $1/CURRENT $2
cp $1/LOG* $2
cp $1/MANIFEST* $2
cp $1/*.log $2
touch $2/LOCK
