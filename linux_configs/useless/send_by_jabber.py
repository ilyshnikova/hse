#!/usr/bin/env python
#-*- coding: utf-8 -*-

import xmpp,sys

xmpp_jid = 'qabs-bamboo@yandex-team.ru' #'tsarkov.jabber@yandex.ru'
xmpp_pwd = 'aCh3ieDi' #'tsarkov.jabber123'

to = sys.argv[1]
msg = sys.argv[2]

jid = xmpp.protocol.JID(xmpp_jid)
client = xmpp.Client(jid.getDomain(),debug=[])
client.connect()
client.auth(jid.getNode(),str(xmpp_pwd),resource='xmpppy')
result = client.send(xmpp.protocol.Message(to,msg))
print result
client.disconnect()
