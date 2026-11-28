// Adapted from the ECE284 Fall 2026 project/core_tb.v template by Mingu Kang.
`timescale 1ns/1ps
module core_tb;
    `include "case_config.vh"
    localparam integer ROWS = 8, COLS = 8, BW = 4, PSUM_BW = 16;
    localparam integer PAD_H = ACT_H + 2*PAD, PAD_W = ACT_W + 2*PAD;
    localparam integer OUT_H = (PAD_H-KERNEL)/STRIDE+1;
    localparam integer OUT_W = (PAD_W-KERNEL)/STRIDE+1;
    localparam integer NIJ = PAD_H*PAD_W, NOUT = OUT_H*OUT_W;
    localparam integer NK = KERNEL*KERNEL, WEIGHT_BASE = 1024;
    localparam integer SUM_BASE = NK*NIJ, RELU_BASE = SUM_BASE+NOUT;
    localparam integer STREAM_DEPTH = 16;
    // SKIP_ZERO: L1 holds only nonzero input words plus an occupancy bitmap.
    localparam integer BITMAP_BASE = WEIGHT_BASE - 64, BITMAP_WORDS = (NIJ + 31) / 32;

    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1;
    reg CEN_xmem = 1, WEN_xmem = 1, CEN_pmem = 1, WEN_pmem = 1;
    reg [10:0] A_xmem = 0, A_pmem = 0;
    reg [31:0] D_xmem = 0;
    reg l0_rd = 0, l0_wr = 0, execute = 0, load = 0, ofifo_rd = 0, acc = 0;
    reg sfp_clear = 0, sfp_relu = 0, pmem_from_sfp = 0;
    wire [33:0] inst;
    wire ofifo_valid, l0_ready, ofifo_full;
    wire [127:0] sfp_out, pmem_read_data;
    wire [31:0] xmem_read_data;
    // Instruction fields from the course template; omit the unused IFIFO.
    assign inst[33] = acc;
    assign inst[32] = CEN_pmem;
    assign inst[31] = WEN_pmem;
    assign inst[30:20] = A_pmem;
    assign inst[19] = CEN_xmem;
    assign inst[18] = WEN_xmem;
    assign inst[17:7] = A_xmem;
    assign inst[6] = ofifo_rd;
    assign inst[5:4] = 2'b00;
    assign inst[3] = l0_rd;
    assign inst[2] = l0_wr;
    assign inst[1] = execute;
    assign inst[0] = load;

    core #(.STREAM_OFIFO(STREAM_OFIFO), .OFIFO_DEPTH(STREAM_DEPTH)) dut(
             .clk(clk), .reset(reset), .inst(inst), .D_xmem(D_xmem),
             .ofifo_valid(ofifo_valid), .sfp_out(sfp_out),
             .sfp_clear(sfp_clear), .sfp_relu(sfp_relu), .pmem_from_sfp(pmem_from_sfp),
             .pmem_read_data(pmem_read_data), .xmem_read_data(xmem_read_data),
             .l0_ready(l0_ready), .ofifo_full(ofifo_full));

    reg [31:0] inputs [0:NIJ-1];
    reg [31:0] weights [0:NK*COLS-1];
    reg [127:0] expected_raw [0:NK*NIJ-1];
    reg [127:0] expected_sum [0:NOUT-1];
    reg [127:0] expected_relu [0:NOUT-1];
    integer cycles = 0, phase = 0, job = 0, stage_start = 0;
    integer metrics_fd, result_fd, k, pos, channel;
    integer checked_words = 0, checks_expected = 0;
    reg [8*128-1:0] result_filename;
    integer fifo_words [0:COLS-1];
    integer fifo_peak [0:COLS-1];
    reg [127:0] observed;
    // Per-kernel issue list: L1 address read into L0, psum SRAM destination, golden index.
    integer issue_addr [0:NIJ-1];
    integer issue_dst [0:NIJ-1];
    integer issue_ref [0:NIJ-1];
    integer issue_count;
    integer compact_index [0:NIJ-1];
    reg [63:0] occupancy;
    reg present [0:NK*NOUT-1];
    integer issued_words, sfu_reads, skipped_passes;
    // SRAM accesses of the work stages; verification reads run as phase 50 or 8.
    integer l1_reads = 0, l1_writes = 0, psum_reads = 0, psum_writes = 0;

    always @(posedge clk)
        if (phase != 50 && phase != 8) begin
            if (!CEN_xmem) begin if (WEN_xmem) l1_reads = l1_reads+1; else l1_writes = l1_writes+1; end
            if (!CEN_pmem) begin if (WEN_pmem) psum_reads = psum_reads+1; else psum_writes = psum_writes+1; end
        end

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (!reset) begin
            if (l0_wr && l0_ready !== 1'b1) $fatal(1, "L0 full at cycle %0d", cycles);
            if (!STREAM_OFIFO && NIJ < 64 && ofifo_full === 1'b1)
                $fatal(1, "unexpected OFIFO full at cycle %0d", cycles);
            if ((acc && sfp_relu) || (acc && sfp_clear) || (sfp_clear && sfp_relu))
                $fatal(1, "conflicting SFP commands");
        end
    end

    // Count producer words and aligned pops in each lane. Reset clears only the live count.
    always @(posedge clk) begin : track_stream_fifo
        integer c, next_words;
        if (STREAM_OFIFO) begin
            for (c=0; c<COLS; c=c+1) begin
                if (reset) begin
                    fifo_words[c] <= 0;
                end else begin
                    next_words = fifo_words[c] + (dut.datapath.array_valid[c] ? 1 : 0)
                                               - ((ofifo_rd && ofifo_valid) ? 1 : 0);
                    if (next_words < 0 || next_words > STREAM_DEPTH)
                        $fatal(1, "stream OFIFO lane %0d exceeded capacity at cycle %0d", c, cycles);
                    fifo_words[c] <= next_words;
                    if (next_words > fifo_peak[c]) fifo_peak[c] <= next_words;
                end
            end
        end
    end

    // Controls change at the falling edge. SRAM/PE registers sample the next rising edge.
    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask

    task begin_stage(input integer number, input integer kernel_index);
        begin
            phase = number;
            stage_start = cycles;
            $display("job=%0d stage=%0d kernel=%0d cycle=%0d", job, number, kernel_index, cycles);
        end
    endtask

    task end_stage(input integer kernel_index);
        begin
            $fdisplay(metrics_fd, "%0d,%0d,%0d,%0d,%0d,%0d", job, phase, kernel_index,
                      stage_start, cycles, cycles-stage_start);
        end
    endtask

    task reset_compute;
        begin
            CEN_xmem = 1; WEN_xmem = 1; CEN_pmem = 1; WEN_pmem = 1;
            l0_rd = 0; l0_wr = 0; execute = 0; load = 0; ofifo_rd = 0;
            acc = 0; sfp_clear = 0; sfp_relu = 0; pmem_from_sfp = 0;
            reset = 1;
            repeat (ROWS+COLS+2) tick();
            reset = 0;
            tick();
        end
    endtask

    task write_xmem(input integer address, input [31:0] data);
        begin
            A_xmem = address; D_xmem = data; CEN_xmem = 0; WEN_xmem = 0;
            tick();
            CEN_xmem = 1; WEN_xmem = 1;
        end
    endtask

    task read_pmem(input integer address, output [127:0] data);
        begin
            A_pmem = address; CEN_pmem = 0; WEN_pmem = 1;
            tick();
            data = pmem_read_data;
            CEN_pmem = 1;
        end
    endtask

    task write_sfp_result(input integer address);
        begin
            pmem_from_sfp = 1;
            A_pmem = address; CEN_pmem = 0; WEN_pmem = 0;
            tick();
            CEN_pmem = 1; WEN_pmem = 1; pmem_from_sfp = 0;
        end
    endtask

    task clear_sfp;
        begin sfp_clear = 1; tick(); sfp_clear = 0; end
    endtask

    // L1 address of the t-th word: contiguous from base, or taken from the issue list.
    function integer l1_address(input integer base, input integer t, input integer listed);
        l1_address = listed ? issue_addr[t] : base + t;
    endfunction

    // Output p of kernel position k reads padded input word window_pos(k, p).
    function integer window_pos(input integer kernel_index, input integer p);
        window_pos = ((p/OUT_W)*STRIDE + kernel_index/KERNEL)*PAD_W
                     + (p%OUT_W)*STRIDE + kernel_index%KERNEL;
    endfunction

    function integer psum_address(input integer kernel_index, input integer p);
        psum_address = SKIP_ZERO ? kernel_index*NOUT + p
                                 : kernel_index*NIJ + window_pos(kernel_index, p);
    endfunction

    task xmem_to_l0(input integer base, input integer count, input integer listed);
        integer t;
        begin
            if (PIPELINE_L0 && count > 0) begin
                l0_wr = 0;
                A_xmem = l1_address(base, 0, listed); CEN_xmem = 0; WEN_xmem = 1;
                tick(); // Prime the SRAM read before writing L0.
                l0_wr = 1;
                for (t=1; t<count; t=t+1) begin
                    A_xmem = l1_address(base, t, listed);
                    tick(); // L0 takes the previous word while SRAM reads the next.
                end
                CEN_xmem = 1;
                tick(); // Write the last word with SRAM read disabled.
                l0_wr = 0;
            end else begin
                for (t=0; t<count; t=t+1) begin
                    A_xmem = l1_address(base, t, listed); CEN_xmem = 0; WEN_xmem = 1;
                    tick();
                    CEN_xmem = 1; l0_wr = 1;
                    tick();
                    l0_wr = 0;
                end
            end
        end
    endtask

    task check_word(input [127:0] actual, input [127:0] wanted,
                    input integer address, input integer stage_number);
        integer c;
        begin
            // Case inequality alone can accept matching X bits in DUT and oracle.
            if ((^actual === 1'bx) || (^wanted === 1'bx))
                $fatal(1, "X/Z in comparison stage=%0d address=%0d", stage_number, address);
            if (actual !== wanted) begin
                for (c=0; c<COLS; c=c+1)
                    if (actual[PSUM_BW*c +: PSUM_BW] !== wanted[PSUM_BW*c +: PSUM_BW])
                        $display("mismatch stage=%0d address=%0d channel=%0d got=%0d expected=%0d",
                                 stage_number, address, c,
                                 $signed(actual[PSUM_BW*c +: PSUM_BW]),
                                 $signed(wanted[PSUM_BW*c +: PSUM_BW]));
                $fatal(1, "data mismatch");
            end
            checked_words = checked_words + 1;
        end
    endtask

    // Read back a psum SRAM word and compare it. The read runs as phase 50, so activity
    // and SRAM-access counts leave it out; its cycle still belongs to the calling stage.
    task verify_pmem(input integer address, input [127:0] wanted, input integer stage_number);
        integer resume;
        reg [127:0] value;
        begin
            resume = phase; phase = 50;
            read_pmem(address, value);
            check_word(value, wanted, address, stage_number);
            phase = resume;
        end
    endtask

    // Stage 1. SKIP_ZERO: the host writes only nonzero input words and an occupancy
    // bitmap; the controller reads the bitmap back from L1 and plans from it alone.
    task load_scratchpad;
        integer t, n, w;
        reg [31:0] bits;
        begin
            begin_stage(1, -1);
            n = 0;
            for (t=0; t<NIJ; t=t+1)
                if (!SKIP_ZERO || inputs[t] !== 32'd0) begin
                    write_xmem(n, inputs[t]); n = n+1;
                end
            for (t=0; t<NK*COLS; t=t+1) write_xmem(WEIGHT_BASE+t, weights[t]);
            if (SKIP_ZERO) begin
                for (w=0; w<BITMAP_WORDS; w=w+1) begin
                    bits = 0;
                    for (t=0; t<32 && w*32+t<NIJ; t=t+1) bits[t] = (inputs[w*32+t] !== 32'd0);
                    write_xmem(BITMAP_BASE+w, bits);
                end
                occupancy = 0;
                for (w=0; w<BITMAP_WORDS; w=w+1) begin
                    A_xmem = BITMAP_BASE+w; CEN_xmem = 0; WEN_xmem = 1;
                    tick();
                    CEN_xmem = 1;
                    occupancy[w*32 +: 32] = xmem_read_data;
                end
                n = 0;
                for (t=0; t<NIJ; t=t+1) begin
                    compact_index[t] = n;
                    if (occupancy[t]) n = n+1;
                end
            end
            end_stage(-1);
        end
    endtask

    // Baseline: issue all NIJ padded words. SKIP_ZERO: issue only the nonzero words
    // inside the valid output window of this kernel position.
    task plan_kernel(input integer kernel_index);
        integer t, p, pos;
        begin
            issue_count = 0;
            if (!SKIP_ZERO) begin
                for (t=0; t<NIJ; t=t+1) begin
                    issue_addr[t] = t;
                    issue_dst[t] = kernel_index*NIJ+t;
                    issue_ref[t] = kernel_index*NIJ+t;
                end
                issue_count = NIJ;
            end else begin
                for (p=0; p<NOUT; p=p+1) begin
                    pos = window_pos(kernel_index, p);
                    present[kernel_index*NOUT+p] = occupancy[pos];
                    if (occupancy[pos]) begin
                        issue_addr[issue_count] = compact_index[pos];
                        issue_dst[issue_count] = kernel_index*NOUT+p;
                        issue_ref[issue_count] = kernel_index*NIJ+pos;
                        issue_count = issue_count+1;
                    end
                end
            end
        end
    endtask

    task load_pe_kernel(input integer kernel_index);
        begin
            reset_compute(); // SRAM contents survive; PE weights/FIFO pointers do not.
            begin_stage(2, kernel_index);
            xmem_to_l0(WEIGHT_BASE+kernel_index*COLS, COLS, 0);
            // L0's read skew and the array's instruction skew both start here.
            l0_rd = 1; load = 1;
            repeat (COLS) tick();
            l0_rd = 0; load = 0;
            repeat (ROWS+COLS+2) tick();
            end_stage(kernel_index);
        end
    endtask

    task load_l0_activations(input integer kernel_index);
        begin
            begin_stage(3, kernel_index);
            xmem_to_l0(0, issue_count, 1);
            end_stage(kernel_index);
        end
    endtask

    task execute_kernel(input integer kernel_index);
        begin
            begin_stage(4, kernel_index);
            l0_rd = 1; execute = 1;
            repeat (issue_count) tick();
            l0_rd = 0; execute = 0;
            repeat (ROWS+COLS+2) tick();
            end_stage(kernel_index);
        end
    endtask

    task store_partials(input integer kernel_index);
        integer t, waited;
        begin
            begin_stage(5, kernel_index);
            pmem_from_sfp = 0;
            for (t=0; t<issue_count; t=t+1) begin
                waited = 0;
                while (ofifo_valid !== 1'b1 && waited < ROWS+COLS+4) begin
                    tick(); waited = waited+1;
                end
                if (ofifo_valid !== 1'b1) $fatal(1, "missing OFIFO word %0d, kernel %0d", t, kernel_index);
                A_pmem = issue_dst[t]; CEN_pmem = 0; WEN_pmem = 0;
                ofifo_rd = 1;
                tick(); // SRAM captures the current head; OFIFO registers the read request.
                CEN_pmem = 1; WEN_pmem = 1; ofifo_rd = 0;
                tick(); // OFIFO consumes that head one clock later.
            end
            if (ofifo_valid !== 1'b0) $fatal(1, "unexpected extra OFIFO results");
            end_stage(kernel_index);
        end
    endtask

    task execute_and_store(input integer kernel_index);
        integer t, stored, c;
        begin
            begin_stage(45, kernel_index);
            stored = 0;
            pmem_from_sfp = 0;
            for (t=0; t<issue_count+ROWS+COLS+2; t=t+1) begin
                l0_rd = (t < issue_count); execute = l0_rd;
                ofifo_rd = (stored < issue_count) && (ofifo_valid === 1'b1);
                CEN_pmem = !ofifo_rd; WEN_pmem = !ofifo_rd;
                if (ofifo_rd) begin
                    A_pmem = issue_dst[stored];
                    stored = stored+1;
                end
                tick(); // The SRAM write and aligned FIFO pop use the same edge.
            end
            l0_rd = 0; execute = 0; ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1;
            if (stored != issue_count) $fatal(1, "stream OFIFO wrote %0d of %0d words", stored, issue_count);
            for (c=0; c<COLS; c=c+1)
                if (fifo_words[c] !== 0) $fatal(1, "stream OFIFO lane %0d still has data", c);
            end_stage(kernel_index);
        end
    endtask

    task check_raw_kernel(input integer kernel_index);
        integer t;
        begin
            begin_stage(50, kernel_index); // Verification time is reported separately.
            for (t=0; t<issue_count; t=t+1)
                verify_pmem(issue_dst[t], expected_raw[issue_ref[t]], 5);
            end_stage(kernel_index);
        end
    endtask

    function present_psum(input integer kernel_index, input integer p);
        present_psum = !SKIP_ZERO || present[kernel_index*NOUT+p];
    endfunction

    task finish_output(input integer p);
        begin
            check_word(sfp_out, expected_sum[p], SUM_BASE+p, 6);
            write_sfp_result(SUM_BASE+p);
            verify_pmem(SUM_BASE+p, expected_sum[p], 6);
        end
    endtask

    task accumulate_outputs;
        integer p, k;
        reg [127:0] value;
        begin
            begin_stage(6, -1);
            for (p=0; p<NOUT; p=p+1) begin
                clear_sfp();
                for (k=0; k<NK; k=k+1)
                    if (present_psum(k, p)) begin
                        read_pmem(psum_address(k, p), value);
                        acc = 1; tick(); acc = 0;
                        sfu_reads = sfu_reads+1;
                    end
                finish_output(p);
            end
            end_stage(-1);
        end
    endtask

    task relu_outputs;
        integer p;
        reg [127:0] value;
        begin
            begin_stage(7, -1);
            for (p=0; p<NOUT; p=p+1) begin
                clear_sfp();
                read_pmem(SUM_BASE+p, value);
                acc = 1; tick(); acc = 0; // Load the stored sum through 0 + sum.
                sfp_relu = 1; tick(); sfp_relu = 0;
                write_sfp_result(RELU_BASE+p);
                verify_pmem(RELU_BASE+p, expected_relu[p], 7);
            end
            end_stage(-1);
        end
    endtask

    // PIPELINE_SFU: the next psum address is issued while the SFU adds the word read on
    // the previous edge, and ReLU follows the stored sum without reloading it.
    // Both the signed sum and the ReLU result are still written back to psum SRAM.
    task accumulate_relu_pipelined;
        integer p, k, n;
        begin
            begin_stage(67, -1);
            for (p=0; p<NOUT; p=p+1) begin
                clear_sfp();
                n = 0;
                for (k=0; k<NK; k=k+1)
                    if (present_psum(k, p)) begin
                        A_pmem = psum_address(k, p); CEN_pmem = 0; WEN_pmem = 1;
                        acc = (n > 0);
                        tick();
                        n = n+1;
                    end
                CEN_pmem = 1;
                if (n > 0) begin acc = 1; tick(); end
                acc = 0;
                sfu_reads = sfu_reads+n;
                finish_output(p);
                sfp_relu = 1; tick(); sfp_relu = 0;
                write_sfp_result(RELU_BASE+p);
                verify_pmem(RELU_BASE+p, expected_relu[p], 7);
            end
            end_stage(-1);
        end
    endtask

    initial begin
        for (channel=0; channel<COLS; channel=channel+1) fifo_peak[channel] = 0;
        if (NIJ > 64 || RELU_BASE+NOUT > 2048 || WEIGHT_BASE+NK*COLS > 2048)
            $fatal(1, "case exceeds memory capacity");
        $readmemb("input.mem", inputs);
        $readmemb("weight.mem", weights);
        $readmemb("expected_raw.mem", expected_raw);
        $readmemb("expected_sum.mem", expected_sum);
        $readmemb("expected_relu.mem", expected_relu);
        // Fail on missing, short or unknown data even when run without Python.
        for (pos=0; pos<NIJ; pos=pos+1)
            if (^inputs[pos] === 1'bx) $fatal(1, "uninitialized input word %0d", pos);
        for (pos=0; pos<NK*COLS; pos=pos+1)
            if (^weights[pos] === 1'bx) $fatal(1, "uninitialized weight word %0d", pos);
        for (pos=0; pos<NK*NIJ; pos=pos+1)
            if (^expected_raw[pos] === 1'bx) $fatal(1, "uninitialized raw oracle %0d", pos);
        for (pos=0; pos<NOUT; pos=pos+1)
            if ((^expected_sum[pos] === 1'bx) || (^expected_relu[pos] === 1'bx))
                $fatal(1, "uninitialized output oracle %0d", pos);
        metrics_fd = $fopen("cycles.csv", "w");
        if (!metrics_fd) $fatal(1, "cannot open cycles.csv");
        $fdisplay(metrics_fd, "job,stage,kernel,start_cycle,end_cycle,cycles");
        if ($test$plusargs("WAVES")) begin
            $dumpfile("core_tb.vcd");
            $dumpvars(0, core_tb);
        end
        @(negedge clk);
        for (job=0; job<REPEATS; job=job+1) begin
            reset_compute();
            issued_words = 0; sfu_reads = 0; skipped_passes = 0;
            l1_reads = 0; l1_writes = 0; psum_reads = 0; psum_writes = 0;
            load_scratchpad();
            for (k=0; k<NK; k=k+1) begin
                plan_kernel(k);
                issued_words = issued_words+issue_count;
                if (issue_count == 0)
                    skipped_passes = skipped_passes+1; // no nonzero word in the window
                else begin
                    load_pe_kernel(k);
                    load_l0_activations(k);
                    if (STREAM_OFIFO)
                        execute_and_store(k);
                    else begin
                        execute_kernel(k);
                        store_partials(k);
                    end
                    check_raw_kernel(k);
                end
            end
            if (PIPELINE_SFU)
                accumulate_relu_pipelined();
            else begin
                accumulate_outputs();
                relu_outputs();
            end
            $display("STATS job=%0d issued_words=%0d sfu_reads=%0d skipped_passes=%0d l1_reads=%0d l1_writes=%0d psum_reads=%0d psum_writes=%0d",
                     job, issued_words, sfu_reads, skipped_passes,
                     l1_reads, l1_writes, psum_reads, psum_writes);
            // One check per issued psum, plus four per output (SFU result, sum, ReLU, final readback).
            checks_expected = checks_expected + issued_words + 4*NOUT;
            phase = 8;
            // $swrite also works with older Icarus runtimes lacking $sformatf.
            $swrite(result_filename, "result_%0d.txt", job);
            result_fd = $fopen(result_filename, "w");
            if (!result_fd) $fatal(1, "cannot open result file");
            for (pos=0; pos<NOUT; pos=pos+1) begin
                read_pmem(RELU_BASE+pos, observed);
                check_word(observed, expected_relu[pos], RELU_BASE+pos, 7);
                $fdisplay(result_fd, "%0128b", observed);
            end
            $fclose(result_fd);
            $display("PASS job=%0d: %0d SRAM output words / %0d scalars", job, NOUT, NOUT*COLS);
        end
        $fclose(metrics_fd);
        if (STREAM_OFIFO) begin
            $write("Stream OFIFO depth=%0d peaks:", STREAM_DEPTH);
            for (channel=0; channel<COLS; channel=channel+1) $write(" %0d", fifo_peak[channel]);
            $display("");
        end
        // run.py recomputes issued_words from the input data, which closes the loop.
        if (checked_words != checks_expected || (!SKIP_ZERO && checks_expected != REPEATS*(NK*NIJ+4*NOUT)))
            $fatal(1, "incomplete verification: %0d word checks", checked_words);
        $display("PASS checks: %0d word comparisons / %0d scalar comparisons", checked_words, checked_words*COLS);
        $display("PASS Step3: seven stages, signed/ReLU SRAM writeback, %0d jobs, total_cycles=%0d",
                 REPEATS, cycles);
        $finish;
    end
    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "global timeout");
    end
endmodule
