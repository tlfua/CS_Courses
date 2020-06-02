#!/usr/bin/env python3

from spanningtreemessage import SpanningTreeMessage
from switchyard.lib.userlib import *
from time import sleep
import struct
import sys
import myswitch_lru

flood_addr = 'ff:ff:ff:ff:ff:ff'
FORWARD, BLOCK = True, False

def mk_stp_pkt(root_id, hops, src, dst):
    # Function to generate STP packets.
    spm = SpanningTreeMessage(root=root_id, hops_to_root=hops)
    Ethernet.add_next_header_class(EtherType.SLOW, SpanningTreeMessage)
    pkt = Ethernet(src=src, dst=dst, ethertype=EtherType.SLOW) + spm
    xbytes = pkt.to_bytes()
    p = Packet(raw=xbytes)
    return p

def mk_pkt(hwsrc, hwdst, ipsrc, ipdst, reply=False):
    # Function to generate normal packets.
    ether = Ethernet(src=hwsrc, dst=hwdst, ethertype=EtherType.IP)
    ippkt = IPv4(src=ipsrc, dst=ipdst, protocol=IPProtocol.ICMP, ttl=32)
    icmppkt = ICMP()
    if reply:
        icmppkt.icmptype = ICMPType.EchoReply
    else:
        icmppkt.icmptype = ICMPType.EchoRequest
    return ether + ippkt + icmppkt

def check_pkt(pkt_str):
    s = set(pkt_str.split(' '))
    if "SpanningTreeMessage" in s:
        return True
    return False

def parse_stp_pkt(pkt_str):
    arr = pkt_str.split(' ')
    # print (p1)
    root, hops = None ,None
    i = 0
    while i < len(arr):
        if arr[i] == "(root:":
            root = arr[i+1]
            # root.pop()
            root = "".join(root[k] for k in range(len(root)-1)) # discard ','
            i += 2
        elif arr[i] == "hops-to-root:":
            hops = arr[i+1]
            hops = "".join(hops[k] for k in range(len(hops)-1)) # discard ')'
            hops = int(hops)
            i += 2
        else:
            i += 1
    if root == None or hops == None:
        print ("Error: do not get both root and hops")
        # sys.exit(0)
    return root, hops


class SwitchData:
    def __init__(self, id, hops_to_root, ports):
        self.id = id
        self.hops = hops_to_root
        self.root = id
        self.port2mode = {}
        for port in ports:
            self.port2mode[port] = FORWARD
        # self.blocked_port_set = set()

    def update(self, new_root, new_hops, input_port):
        if new_root > self.root:
            return False          # do not flood
        elif new_root < self.root:
            self.root = new_root
            self.hops = new_hops + 1
            self.port2mode[input_port] = FORWARD
            return True           # flood
        else:
            if new_hops + 1 < self.hops:
                self.hops = new_hops + 1
                self.port2mode[input_port] = FORWARD
                return True       # flood
            elif new_hops + 1 > self.hops:
                return False      # do not flood
            else:
                self.port2mode[input_port] = BLOCK
                # self.blocked_port_set.add(input_port)
                return False      # do not flood


def main(net):

    my_interfaces = net.interfaces()
    my_ports = set([intf.name for intf in my_interfaces])
    my_addrs = set([intf.ethaddr for intf in my_interfaces])

    lru = myswitch_lru.LRUCache(5)

    # print (min(my_addrs))
    sw_data = SwitchData(str(min(my_addrs)), 0, my_ports)

    for intf in my_interfaces:
        net.send_packet(intf.name, mk_stp_pkt(sw_data.root, sw_data.hops, sw_data.id, flood_addr))

    while True:
        try:
            timestamp, input_port, packet = net.recv_packet()
        except NoPackets:
            log_debug("No packets received!")

            sleep(2)
            if sw_data.root == sw_data.id:
                for intf in my_interfaces:
                    net.send_packet(intf.name, mk_stp_pkt(sw_data.root, sw_data.hops, sw_data.id, flood_addr))
            continue
        except Shutdown:
            log_debug("Received signal for shutdown!")
            return

        pkt_str = str(packet)
        is_stp_pkt = check_pkt(pkt_str)
        if is_stp_pkt:
            new_root, new_hops = parse_stp_pkt(pkt_str)
            to_flood = sw_data.update(new_root, new_hops, input_port)
            if to_flood:
                for intf in my_interfaces:
                    if intf.name != input_port:
                        net.send_packet(intf.name, mk_stp_pkt(sw_data.root, sw_data.hops, sw_data.id, flood_addr))
        else:

            lru.putItem(packet[0].src, input_port)

            if packet[0].dst in my_addrs:
                log_debug ("Packet intended for me")
            else:
                output_port = lru.getPort(packet[0].dst)
                if output_port is not None:
                    net.send_packet(output_port, packet)
                else:
                    for intf in my_interfaces:
                        if intf.name != input_port and sw_data.port2mode[intf.name] == FORWARD:
                            # log_debug ("Flooding packet {} to {}".format(packet, intf.name))
                            net.send_packet(intf.name, packet)
    net.shutdown()