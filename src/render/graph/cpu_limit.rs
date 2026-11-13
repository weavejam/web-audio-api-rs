//! How many threads the parallel executor may use: the CPUs, or for "all
//! cores" the physical cores, this process may run on, capped by its cgroup
//! CPU budget.
//!
//! Whatever needs file IO, allocation or a variable-size OS query (the core
//! topology, the cgroup quota, the logical CPU count, the CPUs of each Windows
//! processor group) is read once by [`prime`], off the render thread, and
//! cached. On the render thread [`usable_cpus`] and [`usable_cores`] only read
//! the current affinity (a syscall or two into stack buffers) and count over
//! the cache: no allocation, lock or file IO. Before [`prime`] they return
//! `usize::MAX` (no limit known); the graph primes whenever it is set to
//! render on more than one thread.
//!
//! `std::thread::available_parallelism` is not used on the render thread: it
//! ignores the process affinity on Windows, and on Linux it reads cgroup
//! files.
//!
//! A second hardware thread on a busy core adds little to a render and slows
//! the thread it shares the core with, so "all cores" counts each physical
//! core with at least one usable CPU once.
//!
//! CPU ids: the kernel's CPU number on Linux/Android; `group * 64 + bit` on
//! Windows (a CPU is a bit of its processor group's affinity mask).
//!
//! Known limits:
//! * Linux/Android: the affinity is read into a fixed `cpu_set_t` (1024 CPUs).
//!   On a machine with more, `sched_getaffinity` fails and every CPU counts,
//!   still capped by the quota.
//! * The cgroup quota is read once; a later change is not followed (the
//!   affinity is, every re-schedule).
//! * Windows: a process whose threads span several processor groups is told
//!   which groups, but not a mask within each, so every active CPU of those
//!   groups counts. This may over-count, never below one group.
//! * macOS/iOS: no affinity and no core topology; every logical CPU counts.

use std::sync::OnceLock;

/// What [`prime`] reads once.
struct Topology {
    /// Logical CPUs of the machine: the limit when the affinity is unknown.
    logical: usize,
    /// The CPU ids of each physical core. Empty if unknown.
    cores: Vec<Vec<usize>>,
    /// By CPU id: whether the CPU belongs to a known core. A usable CPU of no
    /// known core counts as a core of its own.
    in_core: Vec<bool>,
    /// The cgroup CPU budget in whole CPUs (at least 1), if one is set.
    quota: Option<usize>,
    /// By processor group: its active CPUs.
    #[cfg(target_os = "windows")]
    group_masks: Vec<u64>,
}

impl Topology {
    fn cap(&self, n: usize) -> usize {
        self.quota.map_or(n, |q| n.min(q)).max(1)
    }
}

static TOPOLOGY: OnceLock<Topology> = OnceLock::new();

/// Read and cache the topology and quota now, off the render thread.
pub(super) fn prime() {
    let _ = TOPOLOGY.get_or_init(read_topology);
}

fn read_topology() -> Topology {
    let cores = os::core_cpus();
    let mut in_core = Vec::new();
    for &cpu in cores.iter().flatten() {
        if in_core.len() <= cpu {
            in_core.resize(cpu + 1, false);
        }
        in_core[cpu] = true;
    }
    let known = in_core.iter().filter(|&&known| known).count();
    let logical = if known > 0 {
        known
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    };
    #[cfg(target_os = "windows")]
    let group_masks = {
        let mut masks = Vec::new();
        for &cpu in cores.iter().flatten() {
            if masks.len() <= cpu / 64 {
                masks.resize(cpu / 64 + 1, 0u64);
            }
            masks[cpu / 64] |= 1 << (cpu % 64);
        }
        masks
    };
    Topology {
        logical,
        cores,
        in_core,
        quota: os::cpu_quota(),
        #[cfg(target_os = "windows")]
        group_masks,
    }
}

/// Number of CPUs this process may run on, capped by its CPU budget.
pub(super) fn usable_cpus() -> usize {
    let Some(t) = TOPOLOGY.get() else {
        return usize::MAX;
    };
    let cpus = os::allowed(t).map_or(t.logical, |allowed| count_cpus(&allowed));
    t.cap(cpus)
}

/// Number of physical cores among the CPUs this process may run on, capped by
/// its CPU budget.
pub(super) fn usable_cores() -> usize {
    let Some(t) = TOPOLOGY.get() else {
        return usize::MAX;
    };
    let cores = match os::allowed(t) {
        Some(allowed) => count_cores(&t.cores, &t.in_core, &allowed),
        None if t.cores.is_empty() => t.logical,
        None => t.cores.len(),
    };
    t.cap(cores)
}

/// A set of CPU ids, read without allocating.
trait CpuSet {
    fn contains(&self, cpu: usize) -> bool;
    fn for_each(&self, f: impl FnMut(usize));
}

fn count_cpus(set: &impl CpuSet) -> usize {
    let mut n = 0;
    set.for_each(|_| n += 1);
    n
}

/// Physical cores among `allowed`: each known core with at least one allowed
/// CPU, plus each allowed CPU of no known core.
fn count_cores(cores: &[Vec<usize>], in_core: &[bool], allowed: &impl CpuSet) -> usize {
    let mut n = cores
        .iter()
        .filter(|core| core.iter().any(|&cpu| allowed.contains(cpu)))
        .count();
    allowed.for_each(|cpu| {
        if !in_core.get(cpu).copied().unwrap_or(false) {
            n += 1;
        }
    });
    n
}

/// Parse a kernel CPU list such as `0-3,8,10-11`.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for range in list.trim().split(',').filter(|r| !r.is_empty()) {
        match range.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi) = (lo.parse::<usize>().ok()?, hi.parse::<usize>().ok()?);
                if hi < lo || hi - lo > 1 << 16 {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => cpus.push(range.parse().ok()?),
        }
    }
    (!cpus.is_empty()).then_some(cpus)
}

/// Whole CPUs of a `quota / period` budget, at least 1 (as std rounds).
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn whole_cpus(quota: &str, period: &str) -> Option<usize> {
    let quota = quota.trim().parse::<u64>().ok()?;
    let period = period.trim().parse::<u64>().ok()?;
    (period > 0).then(|| ((quota / period) as usize).max(1))
}

/// The CPU budget of this process's cgroup, in whole CPUs: the smallest
/// `cpu.max` (v2) or `cpu.cfs_quota_us / cpu.cfs_period_us` (v1) of its cgroup
/// and every ancestor visible in the mount. `proc_cgroup` and `mountinfo` are
/// `/proc/self/cgroup` and `/proc/self/mountinfo`; `read` reads a file.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn cgroup_quota(
    proc_cgroup: &str,
    mountinfo: &str,
    read: impl Fn(&str) -> Option<String>,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for line in proc_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let v2 = controllers.is_empty();
        if !v2 && !controllers.split(',').any(|c| c == "cpu") {
            continue;
        }
        for mount in mountinfo.lines() {
            let Some((fields, super_fields)) = mount.split_once(" - ") else {
                continue;
            };
            let fields: Vec<&str> = fields.split(' ').collect();
            let super_fields: Vec<&str> = super_fields.split(' ').collect();
            let (Some(&root), Some(&mount_point), Some(&fs_type)) =
                (fields.get(3), fields.get(4), super_fields.first())
            else {
                continue;
            };
            let matches = if v2 {
                fs_type == "cgroup2"
            } else {
                fs_type == "cgroup"
                    && super_fields
                        .get(2)
                        .is_some_and(|options| options.split(',').any(|o| o == "cpu"))
            };
            if !matches {
                continue;
            }
            // The cgroup path relative to the mount; walk from it up to the
            // mount point. A path outside the mount: only the mount point.
            let mut dir = path.strip_prefix(root).unwrap_or("").trim_matches('/');
            loop {
                let base = if dir.is_empty() {
                    mount_point.to_string()
                } else {
                    format!("{mount_point}/{dir}")
                };
                let quota = if v2 {
                    read(&format!("{base}/cpu.max")).and_then(|max| {
                        let (quota, period) = max.trim().split_once(' ')?;
                        whole_cpus(quota, period)
                    })
                } else {
                    read(&format!("{base}/cpu.cfs_quota_us")).and_then(|quota| {
                        whole_cpus(&quota, &read(&format!("{base}/cpu.cfs_period_us"))?)
                    })
                };
                if let Some(quota) = quota {
                    best = Some(best.map_or(quota, |b| b.min(quota)));
                }
                if dir.is_empty() {
                    break;
                }
                dir = dir.rsplit_once('/').map_or("", |(parent, _)| parent);
            }
        }
    }
    best
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod os {
    use super::{CpuSet, Topology};

    pub(super) struct Allowed(libc::cpu_set_t);

    // CPU_SETSIZE is a c_int on glibc, a usize on Android.
    #[allow(trivial_numeric_casts)]
    const SET_SIZE: usize = libc::CPU_SETSIZE as usize;

    impl CpuSet for Allowed {
        fn contains(&self, cpu: usize) -> bool {
            // SAFETY: `cpu` is below CPU_SETSIZE.
            cpu < SET_SIZE && unsafe { libc::CPU_ISSET(cpu, &self.0) }
        }

        fn for_each(&self, mut f: impl FnMut(usize)) {
            (0..SET_SIZE)
                .filter(|&cpu| self.contains(cpu))
                .for_each(&mut f);
        }
    }

    /// The affinity of the calling thread (the render thread).
    pub(super) fn allowed(_t: &Topology) -> Option<Allowed> {
        // SAFETY: `cpu_set_t` is a plain bit set (all-zero is valid) and the
        // pointer and size describe it for the duration of the call.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            (libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0
                && libc::CPU_COUNT(&set) > 0)
                .then_some(Allowed(set))
        }
    }

    fn read(path: &str) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    /// Each core's CPUs, from the sibling list of every possible CPU (sparse
    /// ids included); an offline CPU has none and is in no sibling list.
    pub(super) fn core_cpus() -> Vec<Vec<usize>> {
        let Some(possible) =
            read("/sys/devices/system/cpu/possible").and_then(|l| super::parse_cpu_list(&l))
        else {
            return Vec::new();
        };
        let mut cores = Vec::new();
        for cpu in possible {
            let dir = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            let siblings = read(&format!("{dir}/core_cpus_list"))
                .or_else(|| read(&format!("{dir}/thread_siblings_list")))
                .and_then(|l| super::parse_cpu_list(&l));
            // Each core once, from its lowest CPU.
            if let Some(siblings) = siblings {
                if siblings.iter().min() == Some(&cpu) {
                    cores.push(siblings);
                }
            }
        }
        cores
    }

    pub(super) fn cpu_quota() -> Option<usize> {
        super::cgroup_quota(
            &read("/proc/self/cgroup")?,
            &read("/proc/self/mountinfo")?,
            read,
        )
    }
}

#[cfg(target_os = "windows")]
mod os {
    use super::{CpuSet, Topology};
    use std::ffi::c_void;

    const RELATION_PROCESSOR_CORE: u32 = 0;
    /// More processor groups than Windows supports.
    const MAX_GROUPS: usize = 64;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessAffinityMask(
            process: *mut c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        fn GetProcessGroupAffinity(
            process: *mut c_void,
            group_count: *mut u16,
            group_array: *mut u16,
        ) -> i32;
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            length: *mut u32,
        ) -> i32;
    }

    /// Per processor group the process runs in: its allowed CPUs.
    pub(super) struct Allowed {
        groups: [(u16, u64); MAX_GROUPS],
        len: usize,
    }

    impl CpuSet for Allowed {
        fn contains(&self, cpu: usize) -> bool {
            self.groups[..self.len]
                .iter()
                .any(|&(group, mask)| usize::from(group) == cpu / 64 && mask >> (cpu % 64) & 1 != 0)
        }

        fn for_each(&self, mut f: impl FnMut(usize)) {
            for &(group, mask) in &self.groups[..self.len] {
                (0..64)
                    .filter(|bit| mask >> bit & 1 != 0)
                    .for_each(|bit| f(usize::from(group) * 64 + bit));
            }
        }
    }

    /// The process's groups and its mask within them. One group: the process
    /// affinity mask. Several: Windows gives no per-group mask (the process
    /// mask reads 0), so all active CPUs of each group.
    pub(super) fn allowed(t: &Topology) -> Option<Allowed> {
        // SAFETY: pseudo-handle of the current process.
        let process = unsafe { GetCurrentProcess() };
        let mut groups = [0u16; MAX_GROUPS];
        let mut count = MAX_GROUPS as u16;
        // SAFETY: `groups` holds `count` entries for the duration of the call.
        if unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } == 0 {
            return None;
        }
        let groups = &groups[..usize::from(count).min(MAX_GROUPS)];
        let mut process_mask = 0usize;
        let mut system_mask = 0usize;
        // SAFETY: two valid out pointers for the duration of the call.
        let single =
            unsafe { GetProcessAffinityMask(process, &mut process_mask, &mut system_mask) != 0 }
                && process_mask != 0;
        let mut allowed = Allowed {
            groups: [(0, 0); MAX_GROUPS],
            len: 0,
        };
        if let ([group], true) = (groups, single) {
            allowed.groups[0] = (*group, process_mask as u64);
            allowed.len = 1;
        } else {
            for &group in groups {
                let mask = t.group_masks.get(usize::from(group)).copied().unwrap_or(0);
                if mask != 0 {
                    allowed.groups[allowed.len] = (group, mask);
                    allowed.len += 1;
                }
            }
        }
        (allowed.len > 0).then_some(allowed)
    }

    /// Each core's CPUs, across all processor groups.
    pub(super) fn core_cpus() -> Vec<Vec<usize>> {
        let mut length = 0u32;
        // SAFETY: a null buffer with length 0 only queries the size.
        unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                std::ptr::null_mut(),
                &mut length,
            )
        };
        if length == 0 {
            return Vec::new();
        }
        let mut buffer = vec![0u8; length as usize];
        // SAFETY: `buffer` holds `length` bytes.
        if unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                buffer.as_mut_ptr(),
                &mut length,
            )
        } == 0
        {
            return Vec::new();
        }
        buffer.truncate(length as usize);
        parse_core_records(&buffer)
    }

    /// Parse `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records: `u32
    /// Relationship`, `u32 Size`, then for a core a `PROCESSOR_RELATIONSHIP`
    /// (`u8 Flags, u8 EfficiencyClass, u8 Reserved[20], u16 GroupCount`, then
    /// `GROUP_AFFINITY GroupMask[GroupCount]` at offset 24, each a `usize Mask,
    /// u16 Group, u16 Reserved[3]`).
    pub(super) fn parse_core_records(buffer: &[u8]) -> Vec<Vec<usize>> {
        const WORD: usize = std::mem::size_of::<usize>();
        let bytes = |at: usize, n: usize| buffer.get(at..at + n);
        let u16_at = |at| bytes(at, 2).map(|b| u16::from_ne_bytes([b[0], b[1]]));
        let u32_at = |at| bytes(at, 4).map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]));
        let usize_at = |at| {
            bytes(at, WORD).map(|b| {
                let mut word = [0u8; WORD];
                word.copy_from_slice(b);
                usize::from_ne_bytes(word)
            })
        };
        let mut cores = Vec::new();
        let mut at = 0;
        while let (Some(relationship), Some(size)) = (u32_at(at), u32_at(at + 4)) {
            if size == 0 {
                break;
            }
            if relationship == RELATION_PROCESSOR_CORE {
                let group_count = u16_at(at + 8 + 22).unwrap_or(0);
                let mut cpus = Vec::new();
                for g in 0..usize::from(group_count) {
                    let affinity = at + 8 + 24 + g * (WORD + 8);
                    if let (Some(mask), Some(group)) = (usize_at(affinity), u16_at(affinity + WORD))
                    {
                        cpus.extend(
                            (0..usize::BITS as usize)
                                .filter(|bit| mask >> bit & 1 != 0)
                                .map(|bit| usize::from(group) * 64 + bit),
                        );
                    }
                }
                if !cpus.is_empty() {
                    cores.push(cpus);
                }
            }
            at += size as usize;
        }
        cores
    }

    pub(super) fn cpu_quota() -> Option<usize> {
        // Job object CPU rate limits are not read.
        None
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
mod os {
    use super::{CpuSet, Topology};

    /// No affinity API: never constructed.
    pub(super) enum Allowed {}

    impl CpuSet for Allowed {
        fn contains(&self, _cpu: usize) -> bool {
            match *self {}
        }

        fn for_each(&self, _f: impl FnMut(usize)) {
            match *self {}
        }
    }

    pub(super) fn allowed(_t: &Topology) -> Option<Allowed> {
        None
    }

    pub(super) fn core_cpus() -> Vec<Vec<usize>> {
        Vec::new()
    }

    pub(super) fn cpu_quota() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cpus(Vec<usize>);

    impl CpuSet for Cpus {
        fn contains(&self, cpu: usize) -> bool {
            self.0.contains(&cpu)
        }

        fn for_each(&self, f: impl FnMut(usize)) {
            self.0.iter().copied().for_each(f);
        }
    }

    fn in_core(cores: &[Vec<usize>]) -> Vec<bool> {
        let mut known = vec![false; 256];
        cores.iter().flatten().for_each(|&cpu| known[cpu] = true);
        known
    }

    #[test]
    fn usable_cores_and_cpus() {
        prime();
        let cpus = usable_cpus();
        let cores = usable_cores();
        assert!(cpus >= 1 && cpus < usize::MAX);
        assert!(cores >= 1 && cores <= cpus, "{cores} cores > {cpus} CPUs");
        eprintln!(
            "usable: {cores} cores, {cpus} CPUs of {} logical",
            TOPOLOGY.get().unwrap().logical
        );
    }

    #[test]
    fn a_core_counts_if_any_of_its_cpus_is_allowed() {
        // Cores {0,1} {2,3}, affinity {1,3}: two cores, though neither lowest
        // sibling is allowed.
        let cores = vec![vec![0, 1], vec![2, 3]];
        let known = in_core(&cores);
        assert_eq!(count_cores(&cores, &known, &Cpus(vec![1, 3])), 2);
        assert_eq!(count_cores(&cores, &known, &Cpus(vec![0, 1])), 1);
        assert_eq!(count_cores(&cores, &known, &Cpus(vec![0, 1, 2, 3])), 2);
        // Sparse ids, and a CPU of no known core counts as one.
        let cores = vec![vec![16, 80], vec![17, 81]];
        let known = in_core(&cores);
        assert_eq!(count_cores(&cores, &known, &Cpus(vec![80, 81])), 2);
        assert_eq!(count_cores(&cores, &known, &Cpus(vec![16, 80, 200])), 2);
        assert_eq!(count_cpus(&Cpus(vec![16, 80, 200])), 3);
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("16-17,80-81"), Some(vec![16, 17, 80, 81]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("x"), None);
    }

    fn files(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let entries: Vec<(String, String)> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |path| {
            entries
                .iter()
                .find(|(k, _)| k == path)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn cgroup_v2_quota_is_the_smallest_on_the_path() {
        let mountinfo = "\
22 1 0:21 / /proc rw,nosuid - proc proc rw
30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw,nsdelegate";
        let cgroup = "0::/kubepods/pod1/ctr\n";
        let read = files(&[
            ("/sys/fs/cgroup/kubepods/pod1/ctr/cpu.max", "max 100000\n"),
            ("/sys/fs/cgroup/kubepods/pod1/cpu.max", "250000 100000\n"),
            ("/sys/fs/cgroup/kubepods/cpu.max", "800000 100000\n"),
        ]);
        assert_eq!(cgroup_quota(cgroup, mountinfo, &read), Some(2));
        // No limit anywhere.
        let read = files(&[("/sys/fs/cgroup/kubepods/pod1/ctr/cpu.max", "max 100000")]);
        assert_eq!(cgroup_quota(cgroup, mountinfo, read), None);
        // A container with a cgroup namespace sees its cgroup as the root.
        let read = files(&[("/sys/fs/cgroup/cpu.max", "50000 100000")]);
        assert_eq!(cgroup_quota("0::/\n", mountinfo, read), Some(1));
    }

    #[test]
    fn cgroup_v1_quota() {
        let mountinfo = "\
30 25 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
31 25 0:27 /docker /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct";
        let cgroup = "4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n";
        let read = files(&[
            (
                "/sys/fs/cgroup/cpu,cpuacct/abc/cpu.cfs_quota_us",
                "300000\n",
            ),
            (
                "/sys/fs/cgroup/cpu,cpuacct/abc/cpu.cfs_period_us",
                "100000\n",
            ),
            ("/sys/fs/cgroup/memory/abc/cpu.cfs_quota_us", "100000\n"),
            ("/sys/fs/cgroup/memory/abc/cpu.cfs_period_us", "100000\n"),
        ]);
        assert_eq!(cgroup_quota(cgroup, mountinfo, &read), Some(3));
        let read = files(&[
            ("/sys/fs/cgroup/cpu,cpuacct/abc/cpu.cfs_quota_us", "-1\n"),
            (
                "/sys/fs/cgroup/cpu,cpuacct/abc/cpu.cfs_period_us",
                "100000\n",
            ),
        ]);
        assert_eq!(cgroup_quota(cgroup, mountinfo, read), None);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_core_records_keep_the_group() {
        const WORD: usize = std::mem::size_of::<usize>();
        let record = |group: u16, mask: usize| {
            let size = 8 + 24 + WORD + 8;
            let mut r = vec![0u8; size];
            r[0..4].copy_from_slice(&0u32.to_ne_bytes());
            r[4..8].copy_from_slice(&(size as u32).to_ne_bytes());
            r[8 + 22..8 + 24].copy_from_slice(&1u16.to_ne_bytes());
            r[32..32 + WORD].copy_from_slice(&mask.to_ne_bytes());
            r[32 + WORD..32 + WORD + 2].copy_from_slice(&group.to_ne_bytes());
            r
        };
        let buffer = [record(0, 0b11), record(1, 0b1100)].concat();
        assert_eq!(
            os::parse_core_records(&buffer),
            vec![vec![0, 1], vec![66, 67]]
        );
    }
}
