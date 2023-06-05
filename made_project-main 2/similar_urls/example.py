import recsys

rc = recsys.RecSys(pretrained=False)
rc.train()

print(rc.get_top_urls('https://auto.ru/motorcycle/all/'))