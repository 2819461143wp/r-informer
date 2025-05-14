# 创建logs目录（如果不存在）
mkdir -p logs

# 获取当前时间作为基础时间戳
base_timestamp=$(date +"%Y%m%d_%H%M%S")

# 执行三次
for i in 1 2 3; do
    # 为每次运行创建单独的日志文件
    logfile="logs/informer_qiantangjiang_${base_timestamp}_run${i}.log"

    echo "开始执行第 ${i} 次运行..."

    # 执行Python命令并同时将输出保存到日志文件
    python -u main_informer.py \
        --model informer \
        --data qiantangjiang \
        --freq h \
        --itr 1 \
        --device 0 2>&1 | tee "$logfile"

    echo "第 ${i} 次运行完成。日志已保存到: $logfile"

    # 如果不是最后一次运行，则等待一小段时间
    if [ $i -lt 3 ]; then
        echo "等待10秒后开始下一次运行..."
        sleep 10
    fi
done

echo "所有运行已完成。"