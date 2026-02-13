import ccxt
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==============================================================================
# 1. 核心配置区
# ==============================================================================
CONFIG = {
    'exchange_id': 'binance',
    'proxy': 'http://127.0.0.1:7890', # 确保代理通畅
    'log_file': 'assets_history.csv',
    
    # --- 策略参数 ---
    'risk_free_rate': 0.03,
    'rebalance_threshold': 0.03,
    'min_order_val': 10,
    'fee_rate': 0.001, # 回测手续费千一
    
    # --- BNB 补货 ---
    'bnb_min_value': 5.0,
    'bnb_buy_amount': 15.0,
    
    # --- API 密钥 (交易用) ---
    'REAL': {'api_key': 'atFPRjGtsJpohLCWJFtBZjyGB0a2oH779mm4HIotYIPRE126SESrdhkYfcbDFs7O', 'secret': 'SXuKzzm14Szrw1yscmeAkeeo2hAgZdkj95Nklw0JSWOGATwvr3pa4yhx6meNSxRa'},
    'TEST': {'api_key': '5lfX014FsoR4WS59QYcOi1TKDnlsnjrnNA2OesSN0B3SEBwJUGlz5QavbBCFE0xn', 'secret': '19GZ0m6FYQgTHjuzckFYj4pIMGPUoMzdYTS27kLSlHP3JYIymf79XTFt1SLWz2dr'}
}

# 策略矩阵
STRATEGY_MATRIX = {
    'BEAR':    {'USDT': 0.10, 'BTC': 0.40, 'ETH': 0.20, 'SOL': 0.15, 'BNB': 0.10, 'XRP': 0.05, 'DOGE': 0.00},
    'NEUTRAL': {'USDT': 0.25, 'BTC': 0.30, 'ETH': 0.15, 'SOL': 0.10, 'BNB': 0.10, 'XRP': 0.05, 'DOGE': 0.05},
    'BULL':    {'USDT': 0.45, 'BTC': 0.25, 'ETH': 0.10, 'SOL': 0.05, 'BNB': 0.10, 'XRP': 0.05, 'DOGE': 0.00}
}

class CryptoQuantHybrid:
    def __init__(self, is_sandbox):
        self.is_sandbox = is_sandbox
        self.mode_key = 'TEST' if is_sandbox else 'REAL'
        print(f"\n{'='*70}")
        print(f">>> 混合引擎初始化 | 交易环境: 【{self.mode_key}】 | 回测数据源: 【实盘主网】")
        print(f"{'='*70}")
        
        # 1. 初始化交易账户 (用于下单)
        params = {
            'apiKey': CONFIG[self.mode_key]['api_key'],
            'secret': CONFIG[self.mode_key]['secret'],
            'enableRateLimit': True,
            'timeout': 30000,
            'proxies': {'http': CONFIG['proxy'], 'https': CONFIG['proxy']} if CONFIG['proxy'] else {}
        }
        self.trade_ex = getattr(ccxt, CONFIG['exchange_id'])(params)
        if self.is_sandbox: 
            self.trade_ex.set_sandbox_mode(True)
        
        # 2. 初始化数据预言机 (只用于回测抓取历史数据，强制连实盘)
        # 不需要 Key，只读公共数据
        public_params = {
            'enableRateLimit': True,
            'proxies': {'http': CONFIG['proxy'], 'https': CONFIG['proxy']} if CONFIG['proxy'] else {}
        }
        self.data_ex = getattr(ccxt, CONFIG['exchange_id'])(public_params)
        # 注意：这里绝不设置 sandbox mode，确保抓到的是真实的 BTC 历史
        
        self.trade_ex.load_markets()

    # --------------------------------------------------------------------------
    # 核心修复：使用 data_ex (实盘) 抓取数据，而不是 trade_ex (可能是模拟盘)
    # --------------------------------------------------------------------------
    def fetch_real_history_pagination(self, symbol, days):
        """
        分页抓取实盘历史数据 (突破 1000 条限制)
        """
        timeframe = '1d'
        # 计算开始时间
        now = self.data_ex.milliseconds()
        since = now - days * 24 * 60 * 60 * 1000
        all_ohlcv = []
        
        print(f"    正在从【实盘数据库】抓取 {symbol} 过去 {days} 天数据...", end="")
        
        while True:
            try:
                # 使用 data_ex (实盘)
                ohlcv = self.data_ex.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv: break
                
                new_data_start = ohlcv[0][0]
                new_data_end = ohlcv[-1][0]
                
                # 如果获取的数据已经超过当前时间，停止
                if new_data_start > now: break
                
                all_ohlcv += ohlcv
                
                # 更新游标：最后一条数据的时间 + 1天
                since = new_data_end + 24 * 60 * 60 * 1000
                
                print(".", end="")
                if since >= now: break
                time.sleep(0.1) # 礼貌爬虫
            except Exception as e:
                print(f"抓取中断: {e}")
                break
        
        print(f" 获取到 {len(all_ohlcv)} 条K线")
        
        # 数据清洗
        df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        
        # 去重并排序 (防止API分页重叠)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df['close']

    # --------------------------------------------------------------------------
    # 核心功能：全真历史回测 (Event-Driven)
    # --------------------------------------------------------------------------
    def run_event_driven_backtest(self):
        print(f"\n{'='*70}")
        print(">>> 启动全真历史回测 (数据源: Binance Mainnet)")
        print(f"{'='*70}")
        
        days_map = {'1': 90, '2': 180, '3': 365, '4': 365*4, '5': 365*8}
        choice = input("请选择回测周期: 1.90天 2.半年 3.1年 4.4年(牛熊) 5.8年: ")
        days = days_map.get(choice, 90)
        
        # 1. 准备数据池
        data_pool = {}
        all_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'] 
        
        print("\n>>> 第一步：构建历史时间轴...")
        try:
            # 以 BTC 为基准
            btc_series = self.fetch_real_history_pagination('BTC/USDT', days)
            if len(btc_series) < 10:
                print("❌ 数据获取过少，请检查网络代理！")
                return
            
            data_pool['BTC'] = btc_series
            timeline = btc_series.index
            print(f"    时间范围: {timeline[0].date()} 至 {timeline[-1].date()}")
            
            # 抓取其他币种
            for coin in all_coins:
                if coin == 'BTC': continue
                try:
                    series = self.fetch_real_history_pagination(f"{coin}/USDT", days)
                    # 数据对齐：如果某天没数据(未上市)，填NaN，后续处理
                    data_pool[coin] = series.reindex(timeline) 
                except:
                    print(f"    ⚠️ 无法获取 {coin} 数据，回测将忽略该币种")
                    data_pool[coin] = pd.Series(np.nan, index=timeline)
                    
        except Exception as e:
            print(f"❌ 数据池构建失败: {e}")
            return

        # 2. 初始化回测账户
        print("\n>>> 第二步：开始逐日回演交易...")
        initial_capital = 10000.0
        portfolio = {'USDT': initial_capital}
        for coin in all_coins: portfolio[coin] = 0.0
        
        # 记录器
        history_dates = []
        strategy_nav = []
        benchmark_nav = []
        btc_benchmark_shares = 0
        
        # 3. 时间旅行循环
        for t in timeline:
            # --- A. 获取当天价格快照 ---
            prices = {}
            for coin in all_coins:
                p = data_pool[coin].loc[t]
                # 简单清洗：如果价格是NaN（还没上市），设为0
                prices[coin] = 0 if pd.isna(p) else p
            
            if prices['BTC'] == 0: continue # BTC都没数据，跳过
            
            # --- B. 计算当前净值 ---
            nav = portfolio['USDT']
            for coin in all_coins:
                if prices[coin] > 0:
                    nav += portfolio[coin] * prices[coin]
            
            # 记录历史
            history_dates.append(t)
            strategy_nav.append(nav)
            
            # 设定基准 (第一天全仓买入BTC)
            if btc_benchmark_shares == 0:
                btc_benchmark_shares = initial_capital / prices['BTC']
            benchmark_nav.append(btc_benchmark_shares * prices['BTC'])

            # --- C. 策略核心逻辑 ---
            # 1. 判定牛熊
            state = 'BEAR' if prices['BTC'] < 55000 else ('BULL' if prices['BTC'] > 95000 else 'NEUTRAL')
            target_weights = STRATEGY_MATRIX[state]
            
            # 2. 遍历所有资产进行再平衡
            for coin in all_coins:
                if prices[coin] == 0: continue # 未上市，跳过
                
                target_ratio = target_weights.get(coin, 0)
                target_val = nav * target_ratio
                current_val = portfolio[coin] * prices[coin]
                
                diff = current_val - target_val
                
                # 触发阈值：3%
                if abs(diff) > nav * CONFIG['rebalance_threshold']:
                    trade_amt_usd = abs(diff)
                    
                    if diff > 0: # 卖出 (持仓过重)
                        portfolio[coin] -= trade_amt_usd / prices[coin]
                        portfolio['USDT'] += trade_amt_usd
                    else: # 买入 (持仓过轻)
                        # 检查 USDT 够不够
                        if portfolio['USDT'] > trade_amt_usd:
                            portfolio[coin] += trade_amt_usd / prices[coin]
                            portfolio['USDT'] -= trade_amt_usd
                    
                    # 扣除手续费 (模拟磨损)
                    portfolio['USDT'] -= trade_amt_usd * CONFIG['fee_rate']

        # 4. 生成报告
        self.generate_report(history_dates, strategy_nav, benchmark_nav, initial_capital)

    def generate_report(self, dates, strat, bench, initial):
        # 计算各项指标
        strat_ret = (strat[-1] - initial) / initial * 100
        bench_ret = (bench[-1] - initial) / initial * 100
        
        # 夏普
        s_series = pd.Series(strat)
        pct_change = s_series.pct_change().dropna()
        sharpe = (pct_change.mean() - 0) / pct_change.std() * np.sqrt(365)
        
        # 回撤
        roll_max = s_series.cummax()
        dd = (s_series - roll_max) / roll_max
        max_dd = dd.min() * 100

        print(f"\n{'='*70}")
        print(f"   回测结果报告 ({dates[0].date()} -> {dates[-1].date()})")
        print(f"{'='*70}")
        print(f"初始本金:   ${initial:.2f}")
        print(f"策略净值:   ${strat[-1]:.2f} (收益率: {strat_ret:+.2f}%)")
        print(f"BTC基准:    ${bench[-1]:.2f} (收益率: {bench_ret:+.2f}%)")
        print(f"{'-'*70}")
        print(f"夏普比率:   {sharpe:.2f}")
        print(f"最大回撤:   {max_dd:.2f}%")
        print(f"{'='*70}")
        
        if strat_ret > bench_ret:
            print("🏆 恭喜！策略通过高抛低吸跑赢了死拿BTC。")
        else:
            print("💡 提示：策略跑输了基准。原因可能是大牛市单边上涨，再平衡过早卖飞。")

        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(dates, strat, label='Dynamic Strategy', color='#00b894', linewidth=2)
        plt.plot(dates, bench, label='Buy & Hold BTC', color='gray', linestyle='--', alpha=0.5)
        plt.title(f"Backtest: Strategy ({strat_ret:.0f}%) vs BTC ({bench_ret:.0f}%)")
        plt.xlabel("Year")
        plt.ylabel("Equity (USDT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('backtest_result.png')
        print(">>> 📉 图表已生成: backtest_result.png")

    # --------------------------------------------------------------------------
    # 核心功能 2：实时监控与调仓 (保持原样)
    # --------------------------------------------------------------------------
    def run_live_monitor(self):
        try:
            ticker = self.trade_ex.fetch_ticker('BTC/USDT')
            price = ticker['last']
            balance = self.trade_ex.fetch_balance()
            
            # BNB 补货
            bnb_val = balance['total'].get('BNB', 0) * self.trade_ex.fetch_ticker('BNB/USDT')['last']
            if bnb_val < CONFIG['bnb_min_value']:
                print(f">>> [补货] BNB不足 (${bnb_val:.2f})，正在购买...")
                self.trade_ex.create_market_order('BNB/USDT', 'buy', 0.05)

            # 状态判定
            state = 'BEAR' if price < 55000 else ('BULL' if price > 95000 else 'NEUTRAL')
            targets = STRATEGY_MATRIX[state]
            
            # 计算净值
            total_nav = balance['total'].get('USDT', 0)
            asset_data = {'USDT': {'val': total_nav, 'price': 1}}
            
            # 第一次遍历算总NAV
            for s in targets.keys():
                if s == 'USDT': continue
                p = self.trade_ex.fetch_ticker(f"{s}/USDT")['last']
                q = balance['total'].get(s, 0) or 0
                val = q * p
                asset_data[s] = {'val': val, 'price': p}
                total_nav += val
            
            print(f"\n[监控] BTC: ${price} | 状态: {state} | 总资产: ${total_nav:.2f}")
            print(f"{'-'*90}")
            print(f"{'币种':<6} {'占比':<8} {'持有数量':<12} {'价值($)':<12} {'成本($)':<10} {'盈亏($)':<10}")
            print(f"{'-'*90}")
            
            trades = []
            
            for s, ratio in targets.items():
                if s == 'USDT':
                    curr_ratio = asset_data['USDT']['val'] / total_nav
                    print(f"USDT   {curr_ratio:<8.2%} {'-':<12} {asset_data['USDT']['val']:<12.2f} {'-':<10} {'-'}")
                    continue
                    
                info = asset_data[s]
                curr_ratio = info['val'] / total_nav
                
                # 成本审计
                avg_cost = 0
                try:
                    # 注意：这里用 trade_ex 拉取你的模拟盘成交记录
                    my_trades = self.trade_ex.fetch_my_trades(f"{s}/USDT", limit=100)
                    total_c = sum(t['cost'] for t in my_trades if t['side']=='buy')
                    total_q = sum(t['amount'] for t in my_trades if t['side']=='buy')
                    avg_cost = total_c / total_q if total_q > 0 else 0
                except: pass
                if avg_cost == 0 and self.is_sandbox: avg_cost = info['price']
                
                pnl = info['val'] - (balance['total'].get(s, 0) * avg_cost)
                
                print(f"{s:<6} {curr_ratio:<8.2%} {balance['total'].get(s,0):<12.4f} {info['val']:<12.2f} {avg_cost:<10.2f} {pnl:<10.2f}")
                
                # 调仓判定
                diff = (curr_ratio - ratio) * total_nav
                if abs(diff) > CONFIG['min_order_val'] and abs(curr_ratio - ratio) > CONFIG['rebalance_threshold']:
                    side = 'sell' if diff > 0 else 'buy'
                    trades.append({'symbol': s, 'side': side, 'amt': abs(diff), 'price': info['price']})

            if trades:
                print(f"\n⚠️ 建议调仓 ({len(trades)}):")
                for t in trades:
                    print(f" > {t['side']} {t['symbol']} {t['amt']:.2f} U")
                if input("是否执行? (y/n): ") == 'y':
                    for t in trades:
                        self.trade_ex.create_market_order(f"{t['symbol']}/USDT", t['side'], t['amt']/t['price'])
                    print("执行完毕。")
            else:
                print("\n✅ 比例健康。")

        except Exception as e:
            print(f"监控异常: {e}")

if __name__ == "__main__":
    print("========================================")
    print("   CRYPTO QUANT HYBRID v11.0")
    print("========================================")
    choice = input("1. 模拟盘 (Testnet)  2. 实盘 (Real): ")
    is_sb = True if choice == '1' else False
    
    bot = CryptoQuantHybrid(is_sandbox=is_sb)
    
    while True:
        print("\n1. 实时监控 (Live Monitor)")
        print("2. 历史回测 (Backtest Engine - Real Data)")
        print("3. 退出")
        cmd = input("指令: ")
        if cmd == '1': bot.run_live_monitor()
        elif cmd == '2': bot.run_event_driven_backtest()
        elif cmd == '3': break