from twilio.rest import Client

def sendmsg(bdy):
    account_sid = "AC552ab2a64c07cc59272f2e74d1a0ab69"
    auth_token = "bf78965759d6b5769219e8daa633f4f3"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body=bdy,
    to='whatsapp:+919014046470'
    )