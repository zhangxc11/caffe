#drawlog.sh

P=$(dirname $_)
F=false

while getopts "f" arg
do
    case $arg in
        f)
          F=true
          ;;
        ?)
         ;;
      esac
done

while read -r LOG EPS TITLE; do
    if $F || test $LOG -nt $EPS; then
	python $P/drawlog.py $LOG $EPS "'$TITLE'"
    fi
done < loglist.txt

# cat loglist.txt | while read -r LINE
# do
#     LOG=$(echo $LINE | cut -d ' ' -f 1)
#     EPS=$(echo $LINE | cut -d ' ' -f 2)
#     TITLE=$(echo $LINE | cut -d ' ' -f 3-)
#     if $F || test $LOG -nt $EPS; then
# 	python $P/drawlog.py $LOG $EPS $TITLE
#     fi
# done
