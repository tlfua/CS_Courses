'''
   LRU
'''

from switchyard.lib.userlib import *

class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    # private:
    def _reconnect(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        
    def _insertAfterHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        node.prev.next = node
        node.next.prev = node

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.key2node = dict()
        self.head = Node(-1, -1)    # !
        self.tail = Node(-1, -1)    # !
        self.head.next = self.tail  # !
        self.tail.prev = self.head  # !
        self.val2key = dict()

    def getPort(self, key): # update most recently used one
        """
        :type key: int
        :rtype: int or None
        """
        if key not in self.key2node.keys():
            return None
        node = self.key2node[key]
        self._reconnect(node)
        self._insertAfterHead(node)
        return node.val
        
    def putItem(self, key, value): # may exceed the capacity
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        node = None

        # update val2key
        if value in self.val2key.keys():
            old_key = self.val2key[value]
            del self.val2key[value]
            node = self.key2node[old_key]
            del self.key2node[old_key]
            self._reconnect(node)
        self.val2key[value] = key

        if key in self.key2node.keys(): # key exists, update (key, value) pair
            node = self.key2node[key]
            node.val = value
            self._reconnect(node)
        else:                           # add new list node
            if len(self.key2node) == self.capacity:
                # be careful about the order of (1) & (2)
                del self.key2node[self.tail.prev.key] # (1)
                self._reconnect(self.tail.prev)       # (2)      
            node = Node(key, value)
            self.key2node[key] = node
        self._insertAfterHead(node)
        
def main(net):
    my_interfaces = net.interfaces() 
    my_addrs = set([intf.ethaddr for intf in my_interfaces])

    lru = LRUCache(5)

    while True:
        try:
            timestamp, input_port, packet = net.recv_packet()
        except NoPackets:
            log_debug("No packets received!")
            continue
        except Shutdown:
            log_debug("Received signal for shutdown!")
            return
        
        print (packet[0].src, input_port, packet[0].dst, timestamp)

        lru.putItem(packet[0].src, input_port)
        
        # check if the destination exists in the lru.
        # if it doesn't flood 
        log_debug ("In {} received packet {} on {}".format(net.name, packet, input_port))
        if packet[0].dst in my_addrs:
            log_debug ("Packet intended for me")
        else:
            output_port = lru.getPort(packet[0].dst) 
            if output_port is not None:
                net.send_packet(output_port, packet)
            else:
                for intf in my_interfaces:
                    if input_port != intf.name:
                        log_debug ("Flooding packet {} to {}".format(packet, intf.name))
                        net.send_packet(intf.name, packet)
    net.shutdown()
