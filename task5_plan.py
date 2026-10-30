#!/usr/bin/env python3
"""Task 5 PLAN -- run on the laptop with Python 3, not EV3 MicroPython.

Put task5_nodes_001.csv (outer) and task5_nodes_002.csv (inner) beside this file.
Run: python task5_plan.py
Or: python task5_plan.py outer.csv inner.csv

Builds the 10-node/10-edge graph from recorded operator node visits. No lecture
lengths are inserted. Outer visit index 7 (K, after D2) is excluded because the
operator reported that it was marked on a rounded bypass rather than at K.
BOTH intervals touching that visit are excluded. Valid K->D2 and G1->K
observations remain, so neither edge needs an invented replacement length.

Main cost is accumulated absolute encoder-derived centre travel, in mm,
including line corrections/reversing. It is an empirical contour-travel cost,
NOT surveyed tape centreline length. Repeated observations of each undirected
edge receive equal weight. Signed net advance and observed time are reported
as sensitivity checks; time includes recorded manoeuvres/pauses. Summed edge
times are NOT a measured final S->G1 run time. Raw observations remain in CSV.
Human labels are assumed accurate except the declared excluded reference.
The program plans a route; it does not drive the robot or recognise nodes.
"""
import csv
import heapq
import json
import statistics
import sys
from pathlib import Path

EXPECTED_OUTER = ['S','M','G2','M','N','K','D2','K','G1','K','N','C','S','D4','S']
EXPECTED_INNER = ['S','M','N','C','D1','C','S']
EXCLUDED_OUTER_VISIT = 7


def read_observations(path, route_name, expected):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[-1]['event'] != 'route_complete_manual':
        raise ValueError('%s: run did not finish the visit sequence' % path)
    marks = [r for r in rows if r['event'] in ('node_start','node_manual')]
    if [r['node'] for r in marks] != expected:
        raise ValueError('%s: unexpected node sequence; review the file' % path)
    if [int(r['visit_index']) for r in marks] != list(range(len(expected))):
        raise ValueError('%s: missing or duplicated visit index' % path)
    if any(r['route_name'] != route_name for r in marks):
        raise ValueError('%s: incorrect route selection' % path)
    observations = []
    for a,b in zip(marks, marks[1:]):
        ai,bi = int(a['visit_index']),int(b['visit_index'])
        included = not(route_name == 'outer' and EXCLUDED_OUTER_VISIT in (ai,bi))
        observations.append(dict(
            file=Path(path).name, route_name=route_name,
            from_node=a['node'], to_node=b['node'],
            start_index=ai, end_index=bi,
            start_s=float(a['time_ms'])/1000, end_s=float(b['time_ms'])/1000,
            duration_s=(float(b['time_ms'])-float(a['time_ms']))/1000,
            net_mm=float(b['net_mm'])-float(a['net_mm']),
            centre_travel_mm=float(b['travel_mm'])-float(a['travel_mm']),
            included=included,
            reason='' if included else 'Reported inaccurate K reference after D2'))
    return observations


def build_graph(observations, metric):
    groups = {}
    for row in observations:
        if row['included']:
            key = tuple(sorted((row['from_node'],row['to_node'])))
            groups.setdefault(key,[]).append(float(row[metric]))
    graph, summaries = {}, []
    for (a,b),values in sorted(groups.items()):
        weight = statistics.mean(values)
        if weight <= 0:
            raise ValueError('Nonpositive edge cost: %s-%s' % (a,b))
        graph.setdefault(a,{})[b] = weight
        graph.setdefault(b,{})[a] = weight
        summaries.append(dict(a=a,b=b,mean=weight,count=len(values),
                              minimum=min(values),maximum=max(values)))
    return graph,summaries


def dijkstra(graph, start='S', goal='G1'):
    queue, distances, parents = [(0.,start)],{start:0.},{}
    closed = []
    while queue:
        cost,node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        closed.append(dict(node=node,cost=cost))
        if node == goal:
            route = [goal]
            while route[-1] != start:
                route.append(parents[route[-1]])
            return dict(route=route[::-1],cost=cost,closed=closed)
        for neighbour,weight in graph[node].items():
            if weight < 0:
                raise ValueError('Dijkstra needs nonnegative costs')
            candidate = cost+weight
            if candidate < distances.get(neighbour,float('inf')):
                distances[neighbour],parents[neighbour] = candidate,node
                heapq.heappush(queue,(candidate,neighbour))
    raise ValueError('Goal cannot be reached')


def plan(outer, inner, output_directory):
    observations = (read_observations(outer,'outer',EXPECTED_OUTER) +
                    read_observations(inner,'inner',EXPECTED_INNER))
    result = dict(excluded_outer_visit=EXCLUDED_OUTER_VISIT,
                  interpretation='Empirical contour-travel costs, not surveyed centreline lengths',
                  final_timed_run_recorded=False,metrics={})
    for metric in ('centre_travel_mm','net_mm','duration_s'):
        graph,edges = build_graph(observations,metric)
        if len(graph)!=10 or len(edges)!=10:
            raise ValueError('Incomplete graph; expected 10 nodes and 10 edges')
        found = dijkstra(graph)
        found['alternatives'] = [dict(route=r,cost=sum(graph[a][b] for a,b in zip(r,r[1:])))
                                for r in (['S','M','N','K','G1'],['S','C','N','K','G1'])]
        result['metrics'][metric] = dict(graph=graph,edges=edges,plan=found)
        print(metric, ':', ' -> '.join(found['route']), '= %.3f' % found['cost'])
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    with open(out/'task5_edge_observations.csv','w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(observations[0]))
        writer.writeheader();writer.writerows(observations)
    (out/'task5_measured_plan.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    home=Path(__file__).resolve().parent
    outer=Path(sys.argv[1]) if len(sys.argv)>1 else home/'task5_nodes_001.csv'
    inner=Path(sys.argv[2]) if len(sys.argv)>2 else home/'task5_nodes_002.csv'
    plan(outer,inner,home/'task5_plan_results')
