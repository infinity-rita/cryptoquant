"""
write by rita 2023.2.12
信息聚合器，更新提醒~
"""
import twint

# Configure
c = twint.Config()
c.Username = "realDonaldTrump"
c.Search = "great"

# Run
twint.run.Search(c)
