# pylibz

一个强大的Python工具库，包含多种实用功能。

## 📊 AI模型定价爬虫

使用pyppeteer爬取动态网页的AI模型定价信息，并生成交互式可视化分析。

### 🚀 功能特性

- **动态网页爬取**: 使用pyppeteer爬取JavaScript渲染的动态网页
- **智能数据解析**: 自动识别表格结构，提取AI模型定价信息
- **交互式可视化**: 使用Plotly生成美观的交互式图表
- **多种图表类型**: 包括柱状图、对比图、综合仪表板等
- **数据导出**: 支持CSV格式数据导出
- **中文支持**: 完全支持中文显示

### 📦 安装依赖

```bash
# 使用uv管理依赖（推荐）
uv add pyppeteer pandas plotly

# 或使用pip
pip install pyppeteer pandas plotly
```

### 🔧 使用方法

#### 基础用法

```python
import asyncio
from pylibz.pricing_scraper import PricingScraper

async def main():
    # 创建爬虫实例
    scraper = PricingScraper()
    
    # 爬取编写AI的定价页面
    url = "https://api.bianxie.ai/pricing"
    df = await scraper.run(url)
    
    if df is not None:
        print(f"成功获取 {len(df)} 个AI模型的定价信息")

# 运行爬虫
asyncio.run(main())
```

#### 命令行运行

```bash
# 直接运行脚本
python pylibz/pricing_scraper.py
```

### 📈 生成的文件

运行爬虫后会生成以下文件：

#### 数据文件
- `pylibz/pricing_data.csv` - 原始定价数据（CSV格式）

#### 可视化文件
- `pylibz/pricing_chart.html` - 基础价格对比图
- `pylibz/pricing_comparison.html` - 提示vs补全价格对比图
- `pylibz/pricing_dashboard.html` - 综合分析仪表板

### 🎨 可视化特性

#### 1. 基础价格对比图
- 分别显示提示价格和补全价格的柱状图
- 支持悬浮显示详细信息
- 仅显示前20个模型以保持可读性

#### 2. 价格对比图
- 并排显示提示价格和补全价格
- 便于直观比较两种价格类型
- 支持交互式缩放和筛选

#### 3. 综合分析仪表板
- **价格分布统计**: 直方图显示价格分布情况
- **模型类型分布**: 饼图显示不同AI厂商的占比
- **价格趋势**: 散点图显示价格变化趋势
- **平均价格对比**: 整体平均价格对比

### 📊 数据结构

爬取的数据包含以下字段：

```csv
model_name,prompt_price,completion_price,availability,billing_type,price_text,avg_price
gpt-3.5-turbo,0.5,1.5,可用,按量计费,"提示 $0.5 / 1M tokens 补全 $1.5 / 1M tokens",1.0
```

- `model_name`: AI模型名称
- `prompt_price`: 提示价格（USD/1M tokens）
- `completion_price`: 补全价格（USD/1M tokens）
- `availability`: 可用性状态
- `billing_type`: 计费类型
- `price_text`: 原始价格文本
- `avg_price`: 平均价格

### 🛠️ 自定义配置

#### 修改爬取目标

```python
# 创建爬虫实例
scraper = PricingScraper()

# 爬取其他网站
custom_url = "your-target-website.com/pricing"
df = await scraper.run(custom_url)
```

#### 调整可视化参数

```python
# 修改显示的模型数量
df_custom = df.head(10)  # 只显示前10个模型
scraper.create_bar_chart(df_custom)
```

#### 自定义图表样式

可以在`create_bar_chart`、`create_comparison_chart`等方法中修改：
- 颜色主题
- 图表尺寸
- 字体样式
- 交互功能

### ⚠️ 注意事项

1. **首次运行**: pyppeteer首次运行时会自动下载Chromium浏览器
2. **网络连接**: 需要稳定的网络连接访问目标网站
3. **反爬虫**: 某些网站可能有反爬虫机制，建议适当增加延时
4. **浏览器资源**: 爬虫会启动浏览器实例，确保有足够的系统资源

### 🐛 故障排除

#### 常见问题

1. **中文显示问题**: 已使用Plotly解决，支持完美的中文显示
2. **浏览器启动失败**: 检查系统是否支持headless Chrome
3. **网络超时**: 增加`timeout`参数值
4. **数据提取失败**: 检查目标网站结构是否变化

#### 错误处理

脚本内置了完善的错误处理机制：
- 网络连接失败时使用示例数据
- 浏览器异常时自动清理资源
- 数据解析失败时提供友好提示

### 📄 许可证

MIT License

### 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

### 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件或其他联系方式

---

## 🔗 相关链接

- [pyppeteer文档](https://pyppeteer.github.io/pyppeteer/)
- [Plotly文档](https://plotly.com/python/)
- [Pandas文档](https://pandas.pydata.org/)
