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

    core dut(.clk(clk), .reset(reset), .inst(inst), .D_xmem(D_xmem),
             .ofifo_valid(ofifo_valid), .sfp_out(sfp_out),
             .sfp_clear(sfp_clear), .sfp_relu(sfp_relu), .pmem_from_sfp(pmem_from_sfp),
             .pmem_read_data(pmem_read_data), .l0_ready(l0_ready), .ofifo_full(ofifo_full));

    reg [31:0] inputs [0:NIJ-1];
    reg [31:0] weights [0:NK*COLS-1];
    reg [127:0] expected_raw [0:NK*NIJ-1];
    reg [127:0] expected_sum [0:NOUT-1];
    reg [127:0] expected_relu [0:NOUT-1];
    integer cycles = 0, phase = 0, job = 0, stage_start = 0;
    integer metrics_fd, result_fd, k, pos;
    reg [127:0] observed;

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (!reset) begin
            if (l0_wr && l0_ready !== 1'b1) $fatal(1, "L0 full at cycle %0d", cycles);
            if (NIJ < 64 && ofifo_full === 1'b1) $fatal(1, "unexpected OFIFO full at cycle %0d", cycles);
            if ((acc && sfp_relu) || (acc && sfp_clear) || (sfp_clear && sfp_relu))
                $fatal(1, "conflicting SFP commands");
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

    // First clock latches the SRAM address; the second clock writes that data into L0.
    task xmem_to_l0(input integer base, input integer count);
        integer t;
        begin
            for (t=0; t<count; t=t+1) begin
                A_xmem = base+t; CEN_xmem = 0; WEN_xmem = 1;
                tick();
                CEN_xmem = 1; l0_wr = 1;
                tick();
                l0_wr = 0;
            end
        end
    endtask

    task check_word(input [127:0] actual, input [127:0] wanted,
                    input integer address, input integer stage_number);
        integer c;
        begin
            if (actual !== wanted) begin
                for (c=0; c<COLS; c=c+1)
                    if (actual[PSUM_BW*c +: PSUM_BW] !== wanted[PSUM_BW*c +: PSUM_BW])
                        $display("mismatch stage=%0d address=%0d channel=%0d got=%0d expected=%0d",
                                 stage_number, address, c,
                                 $signed(actual[PSUM_BW*c +: PSUM_BW]),
                                 $signed(wanted[PSUM_BW*c +: PSUM_BW]));
                $fatal(1, "data mismatch");
            end
        end
    endtask

    task load_scratchpad;
        integer t;
        begin
            begin_stage(1, -1);
            for (t=0; t<NIJ; t=t+1) write_xmem(t, inputs[t]);
            for (t=0; t<NK*COLS; t=t+1) write_xmem(WEIGHT_BASE+t, weights[t]);
            end_stage(-1);
        end
    endtask

    task load_pe_kernel(input integer kernel_index);
        begin
            reset_compute(); // SRAM contents survive; PE weights/FIFO pointers do not.
            begin_stage(2, kernel_index);
            xmem_to_l0(WEIGHT_BASE+kernel_index*COLS, COLS);
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
            xmem_to_l0(0, NIJ);
            end_stage(kernel_index);
        end
    endtask

    task execute_kernel(input integer kernel_index);
        begin
            begin_stage(4, kernel_index);
            l0_rd = 1; execute = 1;
            repeat (NIJ) tick();
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
            for (t=0; t<NIJ; t=t+1) begin
                waited = 0;
                while (ofifo_valid !== 1'b1 && waited < ROWS+COLS+4) begin
                    tick(); waited = waited+1;
                end
                if (ofifo_valid !== 1'b1) $fatal(1, "missing OFIFO word %0d, kernel %0d", t, kernel_index);
                A_pmem = kernel_index*NIJ+t; CEN_pmem = 0; WEN_pmem = 0;
                ofifo_rd = 1;
                tick(); // SRAM captures the current head; OFIFO registers the read request.
                CEN_pmem = 1; WEN_pmem = 1; ofifo_rd = 0;
                tick(); // OFIFO consumes that head one clock later.
            end
            if (ofifo_valid !== 1'b0) $fatal(1, "unexpected extra OFIFO results");
            end_stage(kernel_index);
        end
    endtask

    task check_raw_kernel(input integer kernel_index);
        integer t;
        reg [127:0] value;
        begin
            begin_stage(50, kernel_index); // Verification time is reported separately.
            for (t=0; t<NIJ; t=t+1) begin
                read_pmem(kernel_index*NIJ+t, value);
                check_word(value, expected_raw[kernel_index*NIJ+t], kernel_index*NIJ+t, 5);
            end
            end_stage(kernel_index);
        end
    endtask

    task accumulate_outputs;
        integer p, ky, kx, address, kernel_index;
        reg [127:0] value;
        begin
            begin_stage(6, -1);
            for (p=0; p<NOUT; p=p+1) begin
                clear_sfp();
                for (ky=0; ky<KERNEL; ky=ky+1)
                    for (kx=0; kx<KERNEL; kx=kx+1) begin
                        kernel_index = ky*KERNEL+kx;
                        address = kernel_index*NIJ
                                  + ((p/OUT_W)*STRIDE+ky)*PAD_W + (p%OUT_W)*STRIDE+kx;
                        read_pmem(address, value);
                        acc = 1; tick(); acc = 0;
                    end
                check_word(sfp_out, expected_sum[p], SUM_BASE+p, 6);
                write_sfp_result(SUM_BASE+p);
                read_pmem(SUM_BASE+p, value);
                check_word(value, expected_sum[p], SUM_BASE+p, 6);
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
                read_pmem(RELU_BASE+p, value);
                check_word(value, expected_relu[p], RELU_BASE+p, 7);
            end
            end_stage(-1);
        end
    endtask

    initial begin
        if (NIJ > 64 || RELU_BASE+NOUT > 2048 || WEIGHT_BASE+NK*COLS > 2048)
            $fatal(1, "case exceeds memory capacity");
        $readmemb("input.mem", inputs);
        $readmemb("weight.mem", weights);
        $readmemb("expected_raw.mem", expected_raw);
        $readmemb("expected_sum.mem", expected_sum);
        $readmemb("expected_relu.mem", expected_relu);
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
            load_scratchpad();
            for (k=0; k<NK; k=k+1) begin
                load_pe_kernel(k);
                load_l0_activations(k);
                execute_kernel(k);
                store_partials(k);
                check_raw_kernel(k);
            end
            accumulate_outputs();
            relu_outputs();
            phase = 8;
            result_fd = $fopen($sformatf("result_%0d.txt", job), "w");
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
        $display("PASS Step3: seven stages, signed/ReLU SRAM writeback, %0d jobs, total_cycles=%0d",
                 REPEATS, cycles);
        $finish;
    end
    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "global timeout");
    end
endmodule
