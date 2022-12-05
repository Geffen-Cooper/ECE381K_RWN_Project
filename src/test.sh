k=20
for (( c=0; c<$k; c++  ))
do
    if [ $c -gt 15 ]; then
        d=$(($c%16))
        e=$((d+17))
        echo $d $e
    else
        echo $c $(($c+1))
    fi
done