package mkbfv

import (
 "encoding/binary"
 "os"
 "path/filepath"
 "testing"
 "github.com/ldsec/lattigo/v2/ring"
)

func TestFixtureCrossModulusNTT(t *testing.T) {
 dir:=os.Getenv("FPGA_FIXTURE_DIR"); if dir=="" { t.Skip() }
 data,err:=os.ReadFile(filepath.Join(dir,"stage_hoisted_entry_switch0_qp.bin"));if err!=nil{t.Fatal(err)}
 params:=NewParametersFromLiteral(PN14QP439);n:=params.N();q:=params.Q()
 get:=func(d,m int)[]uint64{v:=make([]uint64,n);base:=(d*8+m)*n*8;for i:=range v{v[i]=binary.LittleEndian.Uint64(data[base+i*8:])};return v}
 rq:=params.RingQ();ntt0:=get(0,0);want1:=get(0,1);src:=make([]uint64,n);ring.InvNTT(ntt0,src,n,rq.NttPsiInv[0],rq.NttNInv[0],q[0],rq.MredParams[0])
 src1:=make([]uint64,n);ring.InvNTT(want1,src1,n,rq.NttPsiInv[1],rq.NttNInv[1],q[1],rq.MredParams[1]);raw:=make([]uint64,n);ring.NTT(src,raw,n,rq.NttPsi[1],q[1],rq.MredParams[1],rq.BredParams[1])
 canSrc:=append([]uint64(nil),src...);for i:=range canSrc{canSrc[i]%=q[1]};can:=make([]uint64,n);ring.NTT(canSrc,can,n,rq.NttPsi[1],q[1],rq.MredParams[1],rq.BredParams[1])
 mismatch:=-1;for i:=range src{if src[i]%q[1]!=src1[i]%q[1]{mismatch=i;break}}
 t.Logf("src0=%d src1=%d raw=%d canonical=%d want=%d q0=%d q1=%d sourceMismatch=%d srcM0=%d srcM1=%d",src[0],src1[0],raw[0],can[0],want1[0],q[0],q[1],mismatch,src[mismatch],src1[mismatch])
 if raw[0]!=want1[0]{t.Fatalf("raw recreation mismatch")};if can[0]%q[1]!=want1[0]%q[1]{t.Fatalf("canonical modular mismatch")}
}
