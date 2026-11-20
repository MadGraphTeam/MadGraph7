################################################################################
#
# Copyright (c) 2012 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Program to combine results from channels that have been
     split into multiple jobs. Multi-job channels are identified
     by local file mjobs.dat in the channel directory.
"""
from __future__ import division
from __future__ import absolute_import
import math
import os
import shutil
import re
import logging
import random
import time


try:
    import madgraph
except ImportError:
    import internal.sum_html as sum_html
    import internal.misc as misc
    from internal import InvalidCmd, MadGraph5Error    
else:
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.misc as misc
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR

    
logger = logging.getLogger('madevent.combine_run') # -> stdout

#useful shortcut
pjoin = os.path.join

   
def get_inc_file(path):
    """read the information of fortran inc files and returns
       the definition in a dictionary format.
       This catch PARAMETER (NAME = VALUE)"""
       
    pat = re.compile(r'''PARAMETER\s*\((?P<name>[_\w]*)\s*=\s*(?P<value>[\+\-\ded]*)\)''',
                     re.I)
        
    out = {}   
    for name, value in pat.findall(open(path).read()):
        orig_value = str(value)
        try:
            out[name.lower()] = float(value.replace('d','e'))
        except ValueError:
            out[name] = orig_value
    return out


def get_fortran_str(nb):
    data = '%E' % nb
    nb, power = data.split('E')
    nb = abs(float(nb)) /10
    power = int(power) + 1
    return '%.7fE%+03i' %(nb,power)


def read_abs_weights(path):
    """absolute event weights of a channel/job events.lhe (no LHE header)"""
    weights = []
    with open(path) as fin:
        header = False
        for line in fin:
            if header:
                weights.append(abs(float(line.split()[2])))
                header = False
            elif line.startswith('<event>'):
                header = True
    return weights


def copy_events(fsock, input, scale_wgt, max_wgt):
    """Copy events from separate runs into one file w/ appropriate wgts:
    an event is kept with probability min(1, |w|/max_wgt) (w scaled by scale_wgt),
    with weight max(|w|, max_wgt). Returns the number of events written."""

    nb_evt = 0
    # skip=True suppresses payload lines for a rejected event until next <event>.
    skip = False
    pending_event = False
    # Local bindings to reduce lookups in this per-event hot loop
    rand = random.random
    get_fortran = get_fortran_str
    write = fsock.write
    max_wgt_fortran = get_fortran(max_wgt)
    scale = scale_wgt

    with open(input) as fin:
        for line in fin:
            # The line immediately after <event> contains the event header and raw weight
            if pending_event:
                # A plain split, not split(None, 5): with a maxsplit the
                # last field keeps the trailing newline, and the write
                # below adds another one, leaving a blank line after every
                # event header. Pythia's LHEF reader rejects such an event
                # as corrupt, so the whole file becomes unshowerable.
                data = line.split()
                if len(data) != 6:
                    raise MadGraph5Error("Line after <event> should have 6 entries")

                scaled_wgt = float(data[2]) * scale
                if scaled_wgt < 0:
                    sign = '-'
                    abs_wgt = -scaled_wgt
                else:
                    sign = ''
                    abs_wgt = scaled_wgt
                skip = False
                if abs_wgt < rand() * max_wgt:
                    skip = True
                else:
                    nb_evt += 1
                    if abs_wgt < max_wgt:
                        new_wgt = max_wgt_fortran
                    else:
                        new_wgt = get_fortran(abs_wgt)
                    write("<event>\n")
                    write(' %s  %s%s  %s\n' % ('   '.join(data[:2]),
                                               sign, new_wgt, '  '.join(data[3:])))
                pending_event = False
                continue

            # Defer writing "<event>" until after accept/reject
            if line.startswith("<event>"):
                pending_event = True
                continue

            if not skip:
                write(line)
    return nb_evt


def overweight_max_weight(weights, n_budget, trunc, min_wgt=0.):
    """Smallest maximum weight m >= min_wgt for which the events heavier than m
    carry at most trunc*n_budget units of m: sum_{w>m} w <= trunc*n_budget*m
    (mg7's max_overweight_truncation, with n_budget unweighted events).
    weights: absolute weights; the overweight events keep their weight."""

    ws = sorted(weights, reverse=True)
    if not ws:
        return min_wgt
    c = trunc * n_budget
    best = ws[0]
    above = 0. # sum of the k largest weights
    for k in range(1, len(ws)):
        above += ws[k-1]
        if above > c * best:
            break # m >= above/c only grows with k, beyond the best found
        # exactly k events above m, for m in [ws[k], ws[k-1])
        m = max(ws[k], above / c)
        if m < ws[k-1]:
            best = min(best, m)
    return max(best, min_wgt)


def pool_weights(sources):
    """scaled absolute weights of the sources [(events.lhe path, scale)] and the
    largest unweighting maximum among them (the smallest weight of a source:
    its events lighter than that were already thinned)"""
    weights = []
    min_wgt = 0.
    for path, scale in sources:
        w = [x * scale for x in read_abs_weights(path)]
        if w:
            min_wgt = max(min_wgt, min(w))
            weights += w
    return weights, min_wgt


def write_pool(gdir, sources, max_wgt):
    """unweight the pooled sources at max_wgt (overweights keep their weight)
    into gdir/events.lhe; returns the number of events"""
    tmp = pjoin(gdir, 'events.lhe.pool')
    nb_evt = 0
    with open(tmp, 'w') as fsock:
        for path, scale in sources:
            nb_evt += copy_events(fsock, path, scale, max_wgt)
    os.replace(tmp, pjoin(gdir, 'events.lhe'))
    return nb_evt


class ProcessBudget(object):
    """Overweight budget shared by all the channels of the process
    (enforce_overweight_limit, process mode): sum over the channels of the weight
    of their events above their maximum <= trunc * sum of all the weights. The
    channel maxima are raised greedily, each time where it removes the most
    overweight weight per estimated additional point: a channel whose events
    fall below its target needs (target/events - 1) times its points again."""

    max_candidates = 200 # distinct weights tried per channel, at each end

    def __init__(self, trunc):
        self.trunc = trunc
        self.channels = []

    def add(self, weights, min_wgt, target, points):
        """a channel; returns its index"""
        ws = sorted(weights)  # ascending
        prefix = [0.]
        for w in ws:
            prefix.append(prefix[-1] + w)
        self.channels.append({'w': ws, 'S': prefix, 'm': min_wgt, 'm0': min_wgt,
                              'target': target, 'points': points})
        return len(self.channels) - 1

    @staticmethod
    def above(c, m):
        """(number of events above m, their weight)"""
        import bisect
        i = bisect.bisect_right(c['w'], m)
        return len(c['w']) - i, c['S'][-1] - c['S'][i]

    def events(self, c, m):
        """expected number of events at maximum m"""
        n, ow = self.above(c, m)
        return n + (c['S'][-1] - ow) / m if m > 0 else 0

    def cost(self, c, m):
        nev = self.events(c, m)
        if nev <= 0:
            return float('inf')
        return c['points'] * max(0., c['target'] / nev - 1)

    def best_move(self, i):
        """(cost per removed overweight weight, new maximum) for channel i"""
        import bisect
        c = self.channels[i]
        ow0 = self.above(c, c['m'])[1]
        if ow0 <= 0:
            return None
        cost0 = self.cost(c, c['m'])
        cands = sorted(set(c['w'][bisect.bisect_right(c['w'], c['m']):]))
        if len(cands) > 2 * self.max_candidates:
            cands = cands[:self.max_candidates] + cands[-self.max_candidates:]
        best = None
        for m in cands:
            gain = ow0 - self.above(c, m)[1]
            if gain <= 0:
                continue
            ratio = (self.cost(c, m) - cost0) / gain
            if best is None or ratio < best[0]:
                best = (ratio, m)
        return best

    def solve(self):
        """choose the channel maxima; returns them, in the order of add()"""
        import heapq
        total = sum(c['S'][-1] for c in self.channels)
        ow = [self.above(c, c['m'])[1] for c in self.channels]
        budget = self.trunc * total
        excess = sum(ow) - budget
        heap = []
        version = [0] * len(self.channels) # an entry is stale once its channel moved
        for i in range(len(self.channels)):
            move = self.best_move(i)
            if move:
                heapq.heappush(heap, (move[0], i, 0, move[1]))
        while excess > 0 and heap:
            ratio, i, v, m = heapq.heappop(heap)
            if v != version[i]:
                continue
            c = self.channels[i]
            c['m'] = m
            version[i] += 1
            new_ow = self.above(c, m)[1]
            excess -= ow[i] - new_ow
            ow[i] = new_ow
            move = self.best_move(i)
            if move:
                heapq.heappush(heap, (move[0], i, version[i], move[1]))

        # a raise taken early can become unnecessary after later ones: lower the
        # raised channels again (most costly first) as far as the budget allows
        import bisect
        for i in sorted(range(len(self.channels)),
                        key=lambda i: -self.cost(self.channels[i], self.channels[i]['m'])):
            c = self.channels[i]
            if c['m'] <= c['m0']:
                continue
            lo = bisect.bisect_right(c['w'], c['m0'])
            hi = bisect.bisect_left(c['w'], c['m'])
            for m in [c['m0']] + sorted(set(c['w'][lo:hi])):
                new_ow = self.above(c, m)[1]
                if excess + new_ow - ow[i] <= 0:
                    excess += new_ow - ow[i]
                    ow[i], c['m'] = new_ow, m
                    break
        return [c['m'] for c in self.channels]


class CombineRuns(object):
    
    def __init__(self, me_dir, subproc=None, readonly=False):

        self.me_dir = me_dir
        # Read-only (concurrent) gridpack: metadata (subproc.mg, maxparticles.inc)
        # is read from the shared read-only gridpack (me_dir), but the per-channel
        # P dirs to combine live directly in the worker's cwd (no SubProcesses
        # layer -- see GridPackCmd.prepare_local_dir), and their events/results
        # are written there.
        self.readonly = readonly

        if not subproc:
            subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses',
                                                                 'subproc.mg'))]
        self.subproc = subproc
        maxpart = get_inc_file(pjoin(me_dir, 'Source', 'maxparticles.inc'))
        self.maxparticles = maxpart['max_particles']


        for procname in self.subproc:
            if readonly:
                path = procname
            else:
                path = pjoin(self.me_dir,'SubProcesses', procname)
            channels = self.get_channels(path)
            for channel in channels:
                self.sum_multichannel(channel)
    
    def sum_multichannel(self, channel):
        """Looks in channel to see if there are multiple runs that
        need to be combined. If so combines them into single run"""
       
        start = time.time()
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        if os.path.exists(pjoin(channel, 'multijob.dat')):
            njobs = int(open(pjoin(channel, 'multijob.dat')).read())
        else:
            return
        results = sum_html.Combine_results(channel)
        if njobs:
            logger.debug('find %s multijob in %s' % (njobs, channel))
        else:
            return
        for i in range(njobs):
            if channel.endswith(os.path.pathsep):
                path = channel[:-1] + alphabet[i % 26] + str((i+1)//26) 
            else:
                path = channel + alphabet[i % 26] + str((i+1)//26) 
            results.add_results(name=alphabet[i % 26] + str((i+1)//26) , 
                                filepath=pjoin(path, 'results.dat'))
        
        results.compute_average()
        if results.xsec:
            results.write_results_dat(pjoin(channel, 'results.dat'))
        else:
            return
        ### Adding information in the log file
        fsock = open(pjoin(channel, 'log.txt'), 'a')
        fsock.write('--------------------- Multi run with %s jobs. ---------------------\n'
                    % njobs)
        for r in results:
            
            fsock.write('job %s : %s %s +- %s %s\n' % (r.name, r.xsec, r.axsec,\
                                                       r.xerru, r.nunwgt))  
            
        #Now read in all of the events and write them
        #back out with the appropriate scaled weight
        to_clean = []
        fsock = open(pjoin(channel, 'events.lhe'), 'w')
        #wgt = results.axsec / results.nunwgt
        maxwgt = results.axsec / results.nunwgt 
        tot_nevents, nb_file = 0, 0
        for result in results:  
            #misc.sprint('target:', result.axsec/result.nunwgt)
            #misc.sprint('job %s : %s %s +- %s: %s' % (result.name, result.xsec, result.axsec,\
            #                                           result.xerru, result.nunwgt))
            

            # The events of each job sum to its cross-section and the channel
            # cross-section is their plain average (compute_average), so every
            # job gets the same weight. Weighting by its number of unweighted
            # events instead penalised the jobs whose events contain a large
            # weight: they have fewer unweighted events (the overweight events
            # below keep their weight), and that suppressed exactly the rare
            # large-weight contributions, i.e. the tails of the distributions.
            ratio = 1./len(results)
            i = result.name
            if channel.endswith(os.path.pathsep):
                path = channel[:-1] + i 
            else:
                path = channel + i
            nw = self.copy_events(fsock, pjoin(path,'events.lhe'), ratio, maxwgt)
            tot_nevents += nw
            nb_file += 1
            to_clean.append(path)
        logger.debug("Combined %s file generating %s events for %s (%.1f%%): (%fs) " , nb_file, tot_nevents, channel, 100*tot_nevents/results.nunwgt, time.time()-start)
        for path in to_clean:
            try:
                shutil.rmtree(path)
            except Exception as error:
                pass
            
    get_fortran_str = staticmethod(get_fortran_str)

    def copy_events(self, fsock, input, scale_wgt, max_wgt):
        """ Copy events from separate runs into one file w/ appropriate wgts"""
        return copy_events(fsock, input, scale_wgt, max_wgt)

    def get_channels(self, proc_path):
        """Opens file symfact.dat to determine all channels"""
        sympath = os.path.join(proc_path, 'symfact.dat')
        
        #ncode is number of digits needed for the bw coding
        
        ncode = int(math.log10(3)*(self.maxparticles-3))+1
        channels = []
        for line in open(sympath):
            try:
                xi, j = line.split()
            except Exception:
                break
            xi, j  = float(xi), int(j)
            
            if j > 0:
                k = int(xi) 
                npos = int(math.log10(k))+1
                #Write with correct number of digits
                if xi == k:
                    dirname = 'G%i' % k
                else:
                    dirname = 'G%.{0}f'.format(ncode) % xi
                channels.append(os.path.join(proc_path,dirname))
        return channels
    
        
        
              
