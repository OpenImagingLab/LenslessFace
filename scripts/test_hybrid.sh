CONFIG=$1
CHECKPOINT=$2
WORRDIR=$(echo results/${CONFIG%.*} | sed 's/configs\///g')
mkdir -p $WORRDIR
echo "work dir: $WORRDIR"
SHOWWORRDIR=$(echo visualizations/${CONFIG%.*} | sed 's/configs\///g')
mkdir -p $SHOWWORRDIR
echo "visualizations dir: $SHOWWORRDIR"

python tools/hybrid/test.py \
    $CONFIG \
    --show-dir $SHOWWORRDIR \