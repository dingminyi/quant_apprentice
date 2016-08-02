import DataFetcher
import datetime

if __name__ == '__main__':
    exchangeCD = "XSHE"
    securityID = "002230.XSHE"
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    token = "96a75c3c43df640b4ba477813e58e517dd0df704c608c2060dc984928d8a64ef"
    client = DataFetcher.Client()
    client.init(token)
    code, res = client.getData("/api/market/getTickRTIntraDay.json?field=&startTime=&securityID=" + securityID + "&endTime=")
    f = file(securityID + stamp + ".json", "a+")
    f.write(res)
    f.close()
