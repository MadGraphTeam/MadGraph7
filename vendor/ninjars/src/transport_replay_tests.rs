//! Held full replay through the isolated production MpState, retaining changes.
use super::*;
use serde_json::{Value,json};
use std::io::Write;

fn packed(value:&Value)->Vec<u8> {
    let mut result=Vec::new();
    for value in value.as_array().unwrap() {
        let text=value.as_str().unwrap().as_bytes();
        assert!(text.len()<=RECORD);
        let mut record=[b' ';RECORD];
        record[..text.len()].copy_from_slice(text);
        result.extend(record);
    }
    result
}

#[test]
#[ignore="requires exclusive MP slot and retained 1392-request capture"]
fn all_stored_requests_retain_outputs_after_transport_fix() {
    let input=std::fs::read_to_string(std::env::var("NINJARS_TRANSPORT_REPLAY").unwrap()).unwrap();
    let output=std::env::var("NINJARS_TRANSPORT_REPLAY_OUTPUT").unwrap();
    std::thread::Builder::new().stack_size(128*1024*1024).spawn(move || {
        let mut stream=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap());
        let mut state=MpState::new().unwrap();
        let mut tested=0usize;
        let mut changed_records=0usize;
        let mut changed_fields=0usize;
        let mut failures=0usize;
        for line in input.lines() {
            let record:Value=serde_json::from_str(line).unwrap();
            assert_eq!(record["status"],0);
            let n=record["n"].as_u64().unwrap() as usize;
            let rank=record["rank"].as_u64().unwrap() as usize;
            let (reply,_)=std::sync::mpsc::sync_channel(1);
            let request=MpRequest { selection: PoleSelection::BOTH, context_id:0,n,rank,coefficients:packed(&record["coefficients"]),
                s_matrix:packed(&record["s_matrix"]),momenta:packed(&record["momenta"]),
                masses:packed(&record["masses"]),scale:packed(&record["scale"]),reply };
            assert_eq!(request.coefficients.len(),RECORD*2*ninjars::fast::tensor_layout::ncoeff(rank));
            let expected:Vec<String>=record["output"].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_owned()).collect();
            assert_eq!(expected.len(),8);
            let result=match state.evaluate(&request) {
                Ok((status,output))=> {
                    let actual:Vec<String>=output.chunks_exact(RECORD).map(|row|std::str::from_utf8(row).unwrap().trim().to_owned()).collect();
                    let changed:Vec<usize>=(0..8).filter(|&i|actual.get(i)!=expected.get(i)).collect();
                    if status!=SUCCESS || actual.len()!=8 {failures+=1;}
                    if !changed.is_empty() {changed_records+=1;changed_fields+=changed.len();}
                    json!({"record":tested,"n":n,"rank":rank,"status":status,
                        "expected":expected,"actual":actual,"changed_fields":changed})
                }
                Err(error)=> {failures+=1;json!({"record":tested,"n":n,"rank":rank,"error":error,"expected":expected})}
            };
            serde_json::to_writer(&mut stream,&result).unwrap();stream.write_all(b"\n").unwrap();stream.flush().unwrap();
            tested+=1;
        }
        assert_eq!(tested,1392);
        serde_json::to_writer(&mut stream,&json!({"complete":true,"records":tested,
            "changed_records":changed_records,"changed_fields":changed_fields,"failures":failures})).unwrap();
        stream.write_all(b"\n").unwrap();stream.flush().unwrap();
        assert_eq!(failures,0,"retain and investigate all failure records");
        assert_eq!(changed_records,0,"retain and investigate all changed fields; never rewrite old expected outputs");
    }).unwrap().join().unwrap();
}
