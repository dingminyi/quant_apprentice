import DataFetcher
import datetime

if __name__ == '__main__':
    exchangeCD = "XSHE"
    securityID = "002230.XSHE"
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    token = "6dd2eb167cef9b1622787ea404ced6906bacb288edc55d68408a17a87cebcaf9"
    client = DataFetcher.Client()
    client.init(token)
    code, res = client.getData("/api/market/getTickRTIntraDay.json?field=&startTime=&securityID=" + securityID + "&endTime=")
    f = file(securityID + stamp + ".json", "a+")
    f.write(res)
    f.close()
