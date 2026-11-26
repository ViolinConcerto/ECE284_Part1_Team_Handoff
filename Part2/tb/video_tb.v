`timescale 1ns/1ps
module video_tb;
    `include "video_config.vh"
    localparam integer ROWS=8, COLS=8, PSUM_BW=32;
    localparam integer NPIX=ACT_H*ACT_W, NK=KERNEL*KERNEL;
    localparam integer OUT_H=(ACT_H+2*PAD-KERNEL)/STRIDE+1;
    localparam integer OUT_W=(ACT_W+2*PAD-KERNEL)/STRIDE+1;
    localparam integer NOUT=OUT_H*OUT_W, WEIGHT_BASE=1024;
    localparam integer TMP_BASE=5*NOUT, Y0_BASE=6*NOUT, YI_BASE=7*NOUT;

    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, compute_clear=0, act_signed=0;
    reg CEN_xmem=1, WEN_xmem=1, CEN_pmem=1, WEN_pmem=1;
    reg [10:0] A_xmem=0, A_pmem=0;
    reg [31:0] D_xmem=0;
    reg l0_rd=0, l0_wr=0, execute=0, load=0, ofifo_rd=0, acc=0;
    reg sfp_clear=0, sfp_relu=0, pmem_from_sfp=0;
    reg [4:0] sfp_shift=0;
    wire [33:0] inst={acc,CEN_pmem,WEN_pmem,A_pmem,
                         CEN_xmem,WEN_xmem,A_xmem,ofifo_rd,2'b00,
                         l0_rd,l0_wr,execute,load};
    wire ofifo_valid,l0_ready,ofifo_full;
    wire [255:0] sfp_out,pmem_read_data;
    wire [31:0] weight_load_cycles,weight_load_sessions;
    core dut(.clk(clk),.reset(reset),.compute_clear(compute_clear),.inst(inst),
        .D_xmem(D_xmem),.act_signed(act_signed),.sfp_clear(sfp_clear),
        .sfp_relu(sfp_relu),.sfp_shift(sfp_shift),.pmem_from_sfp(pmem_from_sfp),
        .ofifo_valid(ofifo_valid),.l0_ready(l0_ready),.ofifo_full(ofifo_full),
        .sfp_out(sfp_out),.pmem_read_data(pmem_read_data),
        .weight_load_cycles(weight_load_cycles),.weight_load_sessions(weight_load_sessions));

    reg [31:0] conv_input [0:8*NPIX-1];
    reg [31:0] diff_input [0:5*NPIX-1];
    reg [31:0] weights [0:NK*8-1];
    reg [255:0] raw_conv [0:NK*8*NOUT-1];
    reg [255:0] raw_diff [0:NK*5*NOUT-1];
    reg [255:0] pass_diff [0:5*NOUT-1];
    reg [255:0] frames [0:4*NOUT-1];
    wire [255:0] resident_weights;
    reg [255:0] saved_weights;
    genvar r,c;
    generate for(r=0;r<8;r=r+1) begin: watch_row
        for(c=0;c<8;c=c+1) begin: watch_col
            assign resident_weights[(r*8+c)*4 +: 4] =
                dut.corelet_instance.mac_array_instance.row_num[r].mac_row_instance.col_num[c].mac_tile_instance.b_q;
        end
    end endgenerate

    integer cycles=0,method=0,phase=0,kernel_index=0,pass_index=0,chunk_start=0;
    integer metrics_fd,loads_fd,result_fd,stage_start=0;
    integer logical_passes,tile_executions,count,t,p,frame_index;
    reg [255:0] observed;
    always @(posedge clk) begin
        cycles<=cycles+1;
        if(!reset && !compute_clear) begin
            if(l0_wr && l0_ready!==1'b1) $fatal(1,"L0 overflow");
            if(acc && sfp_clear) $fatal(1,"conflicting SFU controls");
            if(sfp_relu) $fatal(1,"video reconstruction must use raw sums");
        end
    end

    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask
    task begin_stage(input integer number);
        begin phase=number; stage_start=cycles; end
    endtask
    task end_stage;
        begin $fdisplay(metrics_fd,"%0d,%0d,%0d,%0d,%0d,%0d",
            method,phase,kernel_index,pass_index,chunk_start,cycles-stage_start); end
    endtask
    task idle_controls;
        begin
            CEN_xmem=1; WEN_xmem=1; CEN_pmem=1; WEN_pmem=1;
            l0_rd=0; l0_wr=0; execute=0; load=0; ofifo_rd=0;
            acc=0; sfp_clear=0; sfp_relu=0; sfp_shift=0; pmem_from_sfp=0;
        end
    endtask
    task full_reset;
        begin
            begin_stage(0); idle_controls(); compute_clear=0; reset=1;
            repeat(2) tick(); reset=0; tick(); end_stage();
        end
    endtask
    // Flush data/control pipelines and FIFOs without touching resident PE weights.
    task clear_pass;
        begin
            begin_stage(0); idle_controls(); compute_clear=1;
            repeat(2) tick(); compute_clear=0; tick(); end_stage();
            if(resident_weights!==saved_weights) $fatal(1,"compute_clear changed PE weights");
        end
    endtask
    task write_xmem(input integer address,input [31:0] value);
        begin
            A_xmem=address; D_xmem=value; CEN_xmem=0; WEN_xmem=0;
            tick(); CEN_xmem=1; WEN_xmem=1;
        end
    endtask
    // Registered SRAM read followed by L0 capture: two distinct rising edges.
    task xmem_word_to_l0(input integer address);
        begin
            A_xmem=address; CEN_xmem=0; WEN_xmem=1; tick();
            CEN_xmem=1; l0_wr=1; tick(); l0_wr=0;
        end
    endtask
    task read_pmem(input integer address,output [255:0] value);
        begin
            A_pmem=address; CEN_pmem=0; WEN_pmem=1; tick();
            value=pmem_read_data; CEN_pmem=1;
        end
    endtask
    task clear_sfu;
        begin sfp_clear=1; tick(); sfp_clear=0; end
    endtask
    task add_pmem(input integer address,input integer shift);
        reg [255:0] value;
        begin
            read_pmem(address,value); sfp_shift=shift;
            acc=1; tick(); acc=0; sfp_shift=0;
        end
    endtask
    task write_sfu(input integer address);
        begin
            A_pmem=address; CEN_pmem=0; WEN_pmem=0; pmem_from_sfp=1;
            tick(); CEN_pmem=1; WEN_pmem=1; pmem_from_sfp=0;
        end
    endtask
    task check_word(input [255:0] actual,input [255:0] expected,input integer index);
        integer lane;
        begin
            if(actual!==expected) begin
                for(lane=0;lane<8;lane=lane+1)
                    if(actual[lane*32 +:32]!==expected[lane*32 +:32])
                        $display("method=%0d phase=%0d k=%0d pass=%0d p=%0d lane=%0d got=%0d expected=%0d",
                            method,phase,kernel_index,pass_index,index,lane,
                            $signed(actual[lane*32 +:32]),$signed(expected[lane*32 +:32]));
                $fatal(1,"video data mismatch");
            end
        end
    endtask
    task check_resident;
        begin
            if(resident_weights!==saved_weights || weight_load_cycles!==32'd8 || weight_load_sessions!==32'd1)
                $fatal(1,"weights were cleared, changed or reloaded during a kernel session");
        end
    endtask
    task check_loaded_weights;
        integer rr,cc;
        begin
            for(rr=0;rr<8;rr=rr+1)
                for(cc=0;cc<8;cc=cc+1)
                    if(resident_weights[(rr*8+cc)*4 +:4] !==
                       weights[kernel_index*8+cc][rr*4 +:4])
                        $fatal(1,"loaded weight mismatch kernel=%0d row=%0d col=%0d got=%0d expected=%0d",
                            kernel_index,rr,cc,
                            $signed(resident_weights[(rr*8+cc)*4 +:4]),
                            $signed(weights[kernel_index*8+cc][rr*4 +:4]));
        end
    endtask
    task load_kernel;
        integer i;
        begin
            full_reset(); begin_stage(2);
            for(i=0;i<8;i=i+1) xmem_word_to_l0(WEIGHT_BASE+kernel_index*8+i);
            l0_rd=1; load=1; repeat(8) tick(); l0_rd=0; load=0;
            repeat(18) tick();
            check_loaded_weights();
            saved_weights=resident_weights;
            check_resident(); end_stage();
        end
    endtask
    task stage_activations;
        integer i;
        begin
            begin_stage(3);
            // The host/DRAM is emulated by these arrays; both methods use this same transfer policy.
            for(i=0;i<NPIX;i=i+1)
                write_xmem(i,method==0 ? conv_input[pass_index*NPIX+i] : diff_input[pass_index*NPIX+i]);
            write_xmem(NPIX,0); end_stage();
        end
    endtask
    task gather_chunk(input integer first,input integer length);
        integer i,oy,ox,iy,ix,address;
        begin
            begin_stage(4);
            for(i=first;i<first+length;i=i+1) begin
                oy=i/OUT_W; ox=i%OUT_W;
                iy=oy*STRIDE+kernel_index/KERNEL-PAD;
                ix=ox*STRIDE+kernel_index%KERNEL-PAD;
                address=(iy<0 || iy>=ACT_H || ix<0 || ix>=ACT_W) ? NPIX : iy*ACT_W+ix;
                xmem_word_to_l0(address);
            end
            end_stage();
        end
    endtask
    task execute_chunk(input integer length);
        begin
            begin_stage(5); l0_rd=1; execute=1;
            repeat(length) tick(); l0_rd=0; execute=0;
            repeat(18) tick(); end_stage();
        end
    endtask
    task store_chunk(input integer first,input integer length);
        integer i,waited;
        begin
            begin_stage(6); pmem_from_sfp=0;
            for(i=first;i<first+length;i=i+1) begin
                waited=0;
                while(ofifo_valid!==1'b1 && waited<20) begin tick(); waited=waited+1; end
                if(ofifo_valid!==1'b1) $fatal(1,"missing OFIFO output p=%0d",i);
                A_pmem=TMP_BASE+i; CEN_pmem=0; WEN_pmem=0; ofifo_rd=1;
                tick(); CEN_pmem=1; WEN_pmem=1; ofifo_rd=0;
                tick();
            end
            if(ofifo_valid!==1'b0) $fatal(1,"extra OFIFO output after chunk");
            end_stage();
        end
    endtask
    task verify_chunk(input integer first,input integer length);
        integer i;
        reg [255:0] value,wanted;
        begin
            begin_stage(90);
            for(i=first;i<first+length;i=i+1) begin
                read_pmem(TMP_BASE+i,value);
                wanted=method==0 ? raw_conv[(kernel_index*8+pass_index)*NOUT+i]
                                  : raw_diff[(kernel_index*5+pass_index)*NOUT+i];
                check_word(value,wanted,i);
            end
            end_stage(); check_resident();
        end
    endtask
    task accumulate_pass;
        integer i,dest,shift;
        begin
            begin_stage(7);
            for(i=0;i<NOUT;i=i+1) begin
                dest=(method==0 ? pass_index/2 : pass_index)*NOUT+i;
                shift=(method==0 && pass_index%2==0) ? 4 : 0;
                clear_sfu();
                if(kernel_index>0 || (method==0 && pass_index%2==1)) add_pmem(dest,0);
                add_pmem(TMP_BASE+i,shift); write_sfu(dest);
            end
            end_stage(); check_resident();
        end
    endtask
    task export_frame(input integer frame_number,input integer base_address);
        integer i;
        reg [255:0] value;
        begin
            begin_stage(90);
            if(method==0) result_fd=$fopen($sformatf("conventional_frame%0d.txt",frame_number),"w");
            else result_fd=$fopen($sformatf("differential_frame%0d.txt",frame_number),"w");
            if(!result_fd) $fatal(1,"cannot create output file");
            for(i=0;i<NOUT;i=i+1) begin
                read_pmem(base_address+i,value);
                check_word(value,frames[frame_number*NOUT+i],i);
                $fdisplay(result_fd,"%0256b",value);
            end
            $fclose(result_fd); end_stage();
        end
    endtask
    task reconstruct_frames;
        integer i,f;
        reg [255:0] value;
        begin
            begin_stage(90);
            for(f=0;f<5;f=f+1) for(i=0;i<NOUT;i=i+1) begin
                read_pmem(f*NOUT+i,value); check_word(value,pass_diff[f*NOUT+i],i);
            end
            end_stage();
            begin_stage(8);
            for(i=0;i<NOUT;i=i+1) begin
                clear_sfu(); add_pmem(i,4); add_pmem(NOUT+i,0); write_sfu(Y0_BASE+i);
            end
            end_stage(); export_frame(0,Y0_BASE);
            for(f=1;f<4;f=f+1) begin
                begin_stage(8);
                for(i=0;i<NOUT;i=i+1) begin
                    clear_sfu(); add_pmem(Y0_BASE+i,0); add_pmem((f+1)*NOUT+i,0); write_sfu(YI_BASE+i);
                end
                end_stage(); export_frame(f,YI_BASE);
            end
            // Y0 survives all three reconstructions.
            begin_stage(90);
            for(i=0;i<NOUT;i=i+1) begin
                read_pmem(Y0_BASE+i,value); check_word(value,frames[i],i);
            end
            end_stage();
        end
    endtask
    initial begin
        if(CHUNK<1 || CHUNK>64 || NPIX>=1024 || 8*NOUT>2048 || WEIGHT_BASE+NK*8>2048)
            $fatal(1,"unsupported memory/chunk configuration");
        $readmemb("conv_input.mem",conv_input); $readmemb("diff_input.mem",diff_input);
        $readmemb("weight.mem",weights); $readmemb("expected_raw_conv.mem",raw_conv);
        $readmemb("expected_raw_diff.mem",raw_diff); $readmemb("expected_pass_diff.mem",pass_diff);
        $readmemb("expected_frames.mem",frames);
        metrics_fd=$fopen("cycles.csv","w"); loads_fd=$fopen("loads.csv","w");
        if(!metrics_fd || !loads_fd) $fatal(1,"cannot create metrics files");
        $fdisplay(metrics_fd,"method,stage,kernel,pass,chunk,cycles");
        $fdisplay(loads_fd,"method,kernel,load_cycles,load_sessions,passes,tile_executions");
        if($test$plusargs("WAVES")) begin $dumpfile("video_tb.vcd"); $dumpvars(0,video_tb); end
        @(negedge clk);
        for(method=0;method<2;method=method+1) begin
            logical_passes=0; tile_executions=0; kernel_index=-1; pass_index=-1; chunk_start=-1;
            full_reset(); begin_stage(1);
            for(t=0;t<NK*8;t=t+1) write_xmem(WEIGHT_BASE+t,weights[t]);
            end_stage();
            for(kernel_index=0;kernel_index<NK;kernel_index=kernel_index+1) begin
                load_kernel();
                for(pass_index=0;pass_index<(method==0 ? 8 : 5);pass_index=pass_index+1) begin
                    logical_passes=logical_passes+1;
                    act_signed=(method==1 && pass_index>=2);
                    stage_activations();
                    for(chunk_start=0;chunk_start<NOUT;chunk_start=chunk_start+CHUNK) begin
                        count=(NOUT-chunk_start<CHUNK) ? NOUT-chunk_start : CHUNK;
                        clear_pass(); gather_chunk(chunk_start,count); execute_chunk(count);
                        store_chunk(chunk_start,count); verify_chunk(chunk_start,count);
                        tile_executions=tile_executions+1;
                    end
                    accumulate_pass();
                end
                check_resident();
                $fdisplay(loads_fd,"%0d,%0d,%0d,%0d,%0d,%0d",method,kernel_index,
                    weight_load_cycles,weight_load_sessions,(method==0 ? 8 : 5),
                    (method==0 ? 8 : 5)*((NOUT+CHUNK-1)/CHUNK));
            end
            if(method==0) for(frame_index=0;frame_index<4;frame_index=frame_index+1)
                export_frame(frame_index,frame_index*NOUT);
            else reconstruct_frames();
            if(logical_passes!==NK*(method==0 ? 8 : 5)) $fatal(1,"pass count mismatch");
            $display("PASS method=%0d kernel_passes=%0d tile_executions=%0d outputs=%0d",
                method,logical_passes,tile_executions,4*NOUT*8);
        end
        $fclose(metrics_fd); $fclose(loads_fd);
        $display("PASS VIDEO: both methods, exact raw SRAM outputs, resident weights");
        $finish;
    end
    initial begin repeat(2000000) @(posedge clk); $fatal(1,"video watchdog timeout"); end
endmodule
