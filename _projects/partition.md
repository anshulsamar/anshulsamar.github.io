---
layout: narrative
title: Decentralized Partitioning
author: Anshul Samar
date: 2017-12-01
mydate: Dec 2017
---

<a href="https://github.com/anshulsamar/partition"> Partition </a> is a greedy repartitioning system for distributed
graphs. The system operates in a decentralized fashion,
reorganizing vertices and edges across server nodes, without the need
of a master. We operate in an asychronous setting, but assume messages are
eventually delivered. Consensus is achieved through Paxos.  

Built by Naoki Eto and Anshul Samar.

---

With the rise of large distributed data stores, exploiting patterns
and workload history to shard data across nodes is critical for
reducing transaction latency.

Facebook's large social network offers one significant use case. At
Facebook [1], servers receive queries from clients requiring them to get
data from external databases (queries such as: who are my friend's
friends?). Because clients are largely interested in their own
sub-networks, having a single node deal to all queries from a
sub-network can allow nodes to keep common data in cache and reduce
cache misses. Effective partitioning of such a graph - where friends
are vertices and friendships are edges - ensures that queries from
friends are dealt with by the same node. Facebook implemented their
centralized distributed repartitioning algorithm, lowering cache
misses by 50 percent. 

Schism [2], on the other hand, uses repartitioning to
minimize the cost of distributed transactions. The consensus
protocol needed to execute such a transaction "adds network messages,
decreases throughput, increases latency, and potentially leads to
expensive distributed deadlocks." To prove this, the authors conduct a
simple experiment in which clients attempt to read two random rows of
a database sharded across multiple servers. Distributed transactions
are shown to take twice as long to complete, along with a significant
hit to throughput.

To reduce need for distributed transactions, Schism repartitions the
database. Specifically, Schism creates a complete subgraph for
every transaction (each tuple, for example, may be represented by a
vertex). Vertices that represent the same tuple across transactions
are connected by an edge. Edges between vertices of the same tuple are
given weights, with higher weights to those involved in more write
transactions. By partitioning this graph with a min-cut, vertices
sharing transactions get pushed to the same partition, keeping the
number of distributed transactions small. Because replicated vertices
that require updating have higher weight edges, the min-cut prefers to
keep them in the same partition, thus reducing the overhead of writing
to replicated nodes. 

This project attempts to solve the general problem of repartitioning
but unlike Facebook, which uses a master to coordinate repartioning,
Partition is decentralized. While  our underlying algorithm is
similar to Facebook's, we developed it independently before learning
about their solution. We stop the repartitioning after a certain
number of rounds, but it can be run for an arbitrarily long amount of
time. We use Paxos to achieve consensus. 

Our last use case is for large organic networks that have no centralized authority
and are dynamically changing. As distributed ledgers such as
blockchain become unwieldy and are used to record trillions of
transactions, having nodes store only portions of the graph depending
on their geographic location and clusters of transactions may be a
viable solution (for example, more money may be exchanging hands in
one area than across two).

See our final <a
href="https://github.com/anshulsamar/partition/blob/master/final_paper.pdf">report</a>
or github <a
href="https://github.com/anshulsamar/partition">project</a> for more
details. Partition was built for
Stanford's CS244B (Distributed Systems) and CS224W (Analysis of
Networks). Thanks to David Mazieres, Jure Lescovec, Peter Bailis, Seo
Jin Park, Anunay Kulshrestha, and Michael Chang for helpful conversations.

<a href="https://www.youtube.com/watch?v=QHkhyY9atkE">[1]</a> Alon Shalita and Igor Kabiljo. *Using Graph Partitioning in
Distributed Systems Design.* Talk at Scale. 2014.

<a href="http://db.csail.mit.edu/pubs/schism-vldb2010.pdf">[2]</a> Calro Curino, Evan Jones, Yang Zhang, and Sam Madden. *Schism: a
Workload-Driven Approach to Database Replication and Partitioning.*
Proceedings of the VLDB Endowment, Vol. 3, No. 1.