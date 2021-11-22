import simplifyied_network

net = simplifyied_network.network(1, 5, 1, 5e-2)
#net.train([[1,1],[1,2],[2,3],[3,4],[5,5],[8,6],[13,7],[21,8],[34,9],[55,10],[89,11]],144)
net.train([1, 2, 4, 8, 16, 32, 64], 128)
